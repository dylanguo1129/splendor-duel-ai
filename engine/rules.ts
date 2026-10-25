import { LINES } from "./board";
import { CARDS, ROYALS } from "./cards";
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
  BOARD_CELLS,
  DRAW,
  EMPTY,
  GOLD,
  MAX_RESERVED,
  MAX_TOKENS,
  MAX_TURNS,
  NUM_LINES,
  PEARL,
  Phase,
  PYRAMID_SLOTS,
  ROYAL_THRESHOLDS,
  WIN_COLOR_POINTS,
  WIN_CROWNS,
  WIN_POINTS,
  COLOR_JOKER,
} from "./constants";
import { Rng } from "./rng";
import { bagCount, GameState, PlayerState, refillBoard, tokenCount } from "./state";

export const EFFECT_JOKER = 1;
export const EFFECT_TOKEN = 2;
export const EFFECT_STEAL = 3;
export const EFFECT_ROYAL = 4;

export const WIN_REASON_POINTS = 1;
export const WIN_REASON_CROWNS = 2;
export const WIN_REASON_COLOR = 3;
export const WIN_REASON_ANTIPLAY = 4;
export const WIN_REASON_DRAW = 5;

export interface RulesConfig {
  /** BGA anti-playing: hoarding all gold + pearls for 3 consecutive turns without buying loses. */
  antiHoard: boolean;
}
export const DEFAULT_RULES: RulesConfig = { antiHoard: true };

/** Level (1..3) of a pyramid slot. */
export function slotLevel(slot: number): number {
  return slot < 5 ? 1 : slot < 9 ? 2 : 3;
}

export function sumBonus(p: PlayerState): number {
  return p.bonus[1] + p.bonus[2] + p.bonus[3] + p.bonus[4] + p.bonus[5];
}

/** Gold needed to buy the card after bonuses and colored tokens; -1 if joker not allowed. */
export function goldNeeded(p: PlayerState, cardId: number): number {
  const c = CARDS[cardId];
  if (c.joker && sumBonus(p) === 0) return -1;
  let short = c.cost[PEARL] > p.tokens[PEARL] ? c.cost[PEARL] - p.tokens[PEARL] : 0;
  for (let t = 1; t <= 5; t++) {
    const need = c.cost[t] - p.bonus[t];
    if (need > p.tokens[t]) short += need - p.tokens[t];
  }
  return short;
}

export function canAfford(p: PlayerState, cardId: number): boolean {
  const g = goldNeeded(p, cardId);
  return g >= 0 && g <= p.tokens[GOLD];
}

/** Canonical payment: colored tokens first, gold covers the shortfall. Returns spent counts [7]. */
export function canonicalPayment(p: PlayerState, cardId: number): Int8Array {
  const c = CARDS[cardId];
  const pay = new Int8Array(7);
  let gold = 0;
  for (let t = 0; t <= 5; t++) {
    const need = t === PEARL ? c.cost[t] : Math.max(0, c.cost[t] - p.bonus[t]);
    const fromTokens = Math.min(need, p.tokens[t]);
    pay[t] = fromTokens;
    gold += need - fromTokens;
  }
  pay[GOLD] = gold;
  return pay;
}

function boardHasGold(s: GameState): boolean {
  for (let c = 0; c < BOARD_CELLS; c++) if (s.board[c] === GOLD) return true;
  return false;
}

/** Writes legal actions into `out`, returns the count. */
export function legalActions(s: GameState, out: Int16Array | number[]): number {
  let n = 0;
  const p = s.players[s.current];
  switch (s.phase) {
    case Phase.Main: {
      // optional actions
      if (!s.replenished && p.privileges > 0) {
        for (let c = s.lastPrivCell + 1; c < BOARD_CELLS; c++) {
          const t = s.board[c];
          if (t !== EMPTY && t !== GOLD) out[n++] = A_CELL + c;
        }
      }
      const canReplenish = !s.replenished && bagCount(s) > 0;
      if (canReplenish) out[n++] = A_REPLENISH;
      const firstMandatory = n;
      // take a line of 1..3 non-gold tokens
      for (let i = 0; i < NUM_LINES; i++) {
        const cells = LINES[i];
        let ok = true;
        for (let k = 0; k < cells.length; k++) {
          const t = s.board[cells[k]];
          if (t === EMPTY || t === GOLD) {
            ok = false;
            break;
          }
        }
        if (ok) out[n++] = A_TAKE_LINE + i;
      }
      // take gold + reserve
      if (p.reserved.length < MAX_RESERVED && boardHasGold(s)) {
        for (let slot = 0; slot < PYRAMID_SLOTS; slot++) if (s.pyramid[slot] !== EMPTY) out[n++] = A_RESERVE + slot;
        for (let level = 1; level <= 3; level++) if (s.decks[level].length > 0) out[n++] = A_RESERVE + 11 + level;
      }
      // purchase
      for (let slot = 0; slot < PYRAMID_SLOTS; slot++) {
        const id = s.pyramid[slot];
        if (id !== EMPTY && canAfford(p, id)) out[n++] = A_BUY + slot;
      }
      for (let r = 0; r < p.reserved.length; r++) if (canAfford(p, p.reserved[r])) out[n++] = A_BUY + 12 + r;
      if (n === firstMandatory) {
        // No mandatory action: replenishing is forced if possible, otherwise pass.
        n = 0;
        out[n++] = canReplenish ? A_REPLENISH : A_PASS;
      }
      return n;
    }
    case Phase.ReserveGold:
      for (let c = 0; c < BOARD_CELLS; c++) if (s.board[c] === GOLD) out[n++] = A_CELL + c;
      return n;
    case Phase.Joker:
      for (let col = 1; col <= 5; col++) if (p.bonus[col] > 0) out[n++] = A_JOKER + col - 1;
      return n;
    case Phase.TakeToken:
      for (let c = 0; c < BOARD_CELLS; c++) if (s.board[c] === s.takeColor) out[n++] = A_CELL + c;
      return n;
    case Phase.Steal: {
      const o = s.players[1 - s.current];
      for (let t = 0; t <= 5; t++) if (o.tokens[t] > 0) out[n++] = A_STEAL + t;
      return n;
    }
    case Phase.Royal:
      for (const r of s.royalsAvail) out[n++] = A_ROYAL + r;
      return n;
    case Phase.Discard: {
      // Non-decreasing type order (canonical); a type is allowed only if enough
      // tokens of that type or higher remain to finish discarding.
      const need = tokenCount(p) - MAX_TOKENS;
      let atOrAbove = 0;
      for (let t = s.lastDiscard; t <= GOLD; t++) atOrAbove += p.tokens[t];
      for (let t = s.lastDiscard; t <= GOLD; t++) {
        if (p.tokens[t] > 0 && atOrAbove >= need) out[n++] = A_DISCARD + t;
        atOrAbove -= p.tokens[t];
      }
      return n;
    }
    default:
      return 0;
  }
}

const scratch = new Int16Array(512);
export function listActions(s: GameState): number[] {
  const n = legalActions(s, scratch);
  return Array.from(scratch.subarray(0, n));
}

export function givePrivilege(s: GameState, pi: number): void {
  const p = s.players[pi];
  if (s.tablePrivileges > 0) {
    s.tablePrivileges--;
    p.privileges++;
  } else {
    const o = s.players[1 - pi];
    if (o.privileges > 0) {
      o.privileges--;
      p.privileges++;
    }
  }
}

function takeFromBoard(s: GameState, p: PlayerState, cell: number): number {
  const t = s.board[cell];
  p.tokens[t]++;
  s.board[cell] = EMPTY;
  return t;
}

function drawToSlot(s: GameState, slot: number): void {
  const deck = s.decks[slotLevel(slot)];
  s.pyramid[slot] = deck.length > 0 ? deck.pop()! : EMPTY;
}

export function applyAction(s: GameState, a: number, rng: Rng, cfg: RulesConfig = DEFAULT_RULES): void {
  const cur = s.current;
  const p = s.players[cur];
  s.ply++;
  switch (s.phase) {
    case Phase.Main: {
      if (a >= A_CELL && a < A_CELL + BOARD_CELLS) {
        // use a privilege
        const cell = a - A_CELL;
        p.privileges--;
        s.tablePrivileges++;
        takeFromBoard(s, p, cell);
        s.lastPrivCell = cell;
        return;
      }
      if (a === A_REPLENISH) {
        refillBoard(s, rng);
        s.replenished = true;
        givePrivilege(s, 1 - cur);
        return;
      }
      if (a >= A_TAKE_LINE && a < A_TAKE_LINE + NUM_LINES) {
        const cells = LINES[a - A_TAKE_LINE];
        let pearls = 0;
        let first = -1;
        let same = true;
        for (let k = 0; k < cells.length; k++) {
          const t = takeFromBoard(s, p, cells[k]);
          if (t === PEARL) pearls++;
          if (k === 0) first = t;
          else if (t !== first) same = false;
        }
        if (pearls >= 2 || (cells.length === 3 && same)) givePrivilege(s, 1 - cur);
        s.passStreak = 0;
        endOfTurn(s, cfg);
        return;
      }
      if (a >= A_RESERVE && a < A_RESERVE + 15) {
        s.reserveTarget = a - A_RESERVE;
        s.phase = Phase.ReserveGold;
        return;
      }
      if (a >= A_BUY && a < A_BUY + 15) {
        buy(s, a - A_BUY, cfg);
        return;
      }
      if (a === A_PASS) {
        s.passStreak++;
        if (s.passStreak >= 2) {
          s.winner = DRAW;
          s.winReason = WIN_REASON_DRAW;
          s.phase = Phase.GameOver;
          return;
        }
        finishTurn(s, cfg);
        return;
      }
      throw new Error(`illegal action ${a} in Main`);
    }
    case Phase.ReserveGold: {
      takeFromBoard(s, p, a - A_CELL);
      const t = s.reserveTarget;
      if (t < 12) {
        p.reserved.push(s.pyramid[t]);
        p.reservedHidden.push(false);
        drawToSlot(s, t);
      } else {
        p.reserved.push(s.decks[t - 11].pop()!);
        p.reservedHidden.push(true);
      }
      s.reserveTarget = -1;
      s.passStreak = 0;
      endOfTurn(s, cfg);
      return;
    }
    case Phase.Joker: {
      const col = a - A_JOKER + 1;
      const c = CARDS[s.pendingCard];
      s.owner[c.id] = col;
      p.bonus[col] += c.bonus;
      p.colorPoints[col] += c.points;
      s.pendingCard = -1;
      s.effects.shift();
      continueEffects(s, cfg);
      return;
    }
    case Phase.TakeToken:
      takeFromBoard(s, p, a - A_CELL);
      s.effects.shift();
      continueEffects(s, cfg);
      return;
    case Phase.Steal: {
      const t = a - A_STEAL;
      const o = s.players[1 - cur];
      o.tokens[t]--;
      p.tokens[t]++;
      s.effects.shift();
      continueEffects(s, cfg);
      return;
    }
    case Phase.Royal: {
      const r = ROYALS[a - A_ROYAL];
      s.royalsAvail.splice(s.royalsAvail.indexOf(r.id), 1);
      p.royals.push(r.id);
      p.points += r.points;
      s.effects.shift();
      if (r.privilege) givePrivilege(s, cur);
      if (r.again) s.extraTurn = true;
      if (r.steal) s.effects.unshift(EFFECT_STEAL);
      continueEffects(s, cfg);
      return;
    }
    case Phase.Discard: {
      const t = a - A_DISCARD;
      p.tokens[t]--;
      s.bag[t]++;
      s.lastDiscard = t;
      if (tokenCount(p) <= MAX_TOKENS) finishTurn(s, cfg);
      return;
    }
    default:
      throw new Error(`no actions in phase ${s.phase}`);
  }
}

function buy(s: GameState, src: number, cfg: RulesConfig): void {
  const cur = s.current;
  const p = s.players[cur];
  let id: number;
  if (src < 12) {
    id = s.pyramid[src];
    drawToSlot(s, src);
  } else {
    const r = src - 12;
    id = p.reserved[r];
    p.reserved.splice(r, 1);
    p.reservedHidden.splice(r, 1);
  }
  const c = CARDS[id];
  const pay = canonicalPayment(p, id);
  for (let t = 0; t < 7; t++) {
    p.tokens[t] -= pay[t];
    s.bag[t] += pay[t];
  }
  p.numCards++;
  p.points += c.points;
  p.crowns += c.crowns;
  s.boughtThisTurn = true;
  s.passStreak = 0;
  s.effects.length = 0;
  if (c.joker) {
    s.pendingCard = id;
    s.effects.push(EFFECT_JOKER);
  } else {
    s.owner[id] = c.color;
    if (c.color !== COLOR_JOKER) {
      p.bonus[c.color] += c.bonus;
      p.colorPoints[c.color] += c.points;
    }
  }
  if (c.token) {
    s.takeColor = c.color;
    s.effects.push(EFFECT_TOKEN);
  }
  if (c.steal) s.effects.push(EFFECT_STEAL);
  if (c.privilege) givePrivilege(s, cur);
  if (c.again) s.extraTurn = true;
  while (p.royalsEarned < ROYAL_THRESHOLDS.length && p.crowns >= ROYAL_THRESHOLDS[p.royalsEarned]) {
    p.royalsEarned++;
    s.effects.push(EFFECT_ROYAL);
  }
  continueEffects(s, cfg);
}

/** Enter the phase of the next pending effect that has a choice, or end the turn. */
function continueEffects(s: GameState, cfg: RulesConfig): void {
  while (s.effects.length > 0) {
    const e = s.effects[0];
    if (e === EFFECT_JOKER) {
      s.phase = Phase.Joker;
      return;
    }
    if (e === EFFECT_TOKEN) {
      for (let c = 0; c < BOARD_CELLS; c++)
        if (s.board[c] === s.takeColor) {
          s.phase = Phase.TakeToken;
          return;
        }
    } else if (e === EFFECT_STEAL) {
      const o = s.players[1 - s.current];
      for (let t = 0; t <= 5; t++)
        if (o.tokens[t] > 0) {
          s.phase = Phase.Steal;
          return;
        }
    } else if (e === EFFECT_ROYAL) {
      if (s.royalsAvail.length > 0) {
        s.phase = Phase.Royal;
        return;
      }
    }
    s.effects.shift(); // effect with nothing to choose: skipped
  }
  endOfTurn(s, cfg);
}

function endOfTurn(s: GameState, cfg: RulesConfig): void {
  const p = s.players[s.current];
  if (tokenCount(p) > MAX_TOKENS) {
    s.phase = Phase.Discard;
    s.lastDiscard = 0;
    return;
  }
  finishTurn(s, cfg);
}

export function checkWin(p: PlayerState): number {
  if (p.points >= WIN_POINTS) return WIN_REASON_POINTS;
  if (p.crowns >= WIN_CROWNS) return WIN_REASON_CROWNS;
  for (let col = 1; col <= 5; col++) if (p.colorPoints[col] >= WIN_COLOR_POINTS) return WIN_REASON_COLOR;
  return 0;
}

function finishTurn(s: GameState, cfg: RulesConfig): void {
  const cur = s.current;
  const p = s.players[cur];
  const reason = checkWin(p);
  if (reason) {
    s.winner = cur;
    s.winReason = reason;
    s.phase = Phase.GameOver;
    return;
  }
  if (cfg.antiHoard) {
    if (p.tokens[GOLD] === 3 && p.tokens[PEARL] === 2 && !s.boughtThisTurn) s.hoard[cur]++;
    else s.hoard[cur] = 0;
    if (s.hoard[cur] >= 3) {
      s.winner = 1 - cur;
      s.winReason = WIN_REASON_ANTIPLAY;
      s.phase = Phase.GameOver;
      return;
    }
  }
  startTurn(s, s.extraTurn ? cur : 1 - cur);
}

export function startTurn(s: GameState, player: number): void {
  s.current = player;
  s.turn++;
  s.phase = Phase.Main;
  s.replenished = false;
  s.lastPrivCell = -1;
  s.lastDiscard = 0;
  s.reserveTarget = -1;
  s.pendingCard = -1;
  s.effects.length = 0;
  s.takeColor = -1;
  s.extraTurn = false;
  s.boughtThisTurn = false;
  if (s.turn >= MAX_TURNS) {
    s.winner = DRAW;
    s.winReason = WIN_REASON_DRAW;
    s.phase = Phase.GameOver;
  }
}

export function isTerminal(s: GameState): boolean {
  return s.phase === Phase.GameOver;
}
