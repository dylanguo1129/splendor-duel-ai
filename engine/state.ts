import { SPIRAL } from "./board";
import { LEVEL_CARD_IDS, NUM_CARDS, ROYALS } from "./cards";
import {
  BOARD_CELLS,
  EMPTY,
  GOLD,
  LEVEL_SLOT_COUNT,
  LEVEL_SLOT_START,
  NUM_TOKEN_TYPES,
  PEARL,
  Phase,
  PYRAMID_SLOTS,
  TOTAL_PRIVILEGES,
} from "./constants";
import { Rng } from "./rng";

export interface PlayerState {
  tokens: Int8Array; // [7] by token type
  bonus: Int8Array; // [6] bonus per gem color (index 1..5; 0 unused)
  colorPoints: Int8Array; // [6] prestige per color column (index 1..5)
  points: number; // total prestige (cards + royals)
  crowns: number;
  privileges: number;
  reserved: number[]; // card ids, in reservation order
  reservedHidden: boolean[]; // reserved blind from a deck (unknown to the opponent)
  royals: number[]; // royal ids
  royalsEarned: number; // crown thresholds already rewarded (0..2)
  numCards: number; // purchased jewel cards
}

export interface GameState {
  board: Int8Array; // [25] token type or EMPTY
  bag: Int8Array; // [7]
  players: [PlayerState, PlayerState];
  tablePrivileges: number;
  pyramid: Int8Array; // [12] card id or EMPTY
  decks: number[][]; // [4], index = level; top of deck = last element
  royalsAvail: number[];
  owner: Int8Array; // [67] -1 unowned, else column: 1..5, 9 colorless (purchased cards only)
  current: number; // 0 | 1
  firstPlayer: number;
  phase: Phase;
  // --- turn bookkeeping ---
  replenished: boolean; // replenished this turn: no further privilege use
  lastPrivCell: number; // privilege picks in increasing cell order (canonical)
  lastDiscard: number; // discards in non-decreasing type order (canonical)
  reserveTarget: number; // pending reserve target while choosing the gold cell
  pendingCard: number; // card awaiting joker color / effects
  effects: number[]; // queued effects (see rules.ts EFFECT_*)
  takeColor: number; // color for TAKE_TOKEN ability
  extraTurn: boolean;
  boughtThisTurn: boolean;
  passStreak: number;
  hoard: Int8Array; // [2] consecutive turn starts holding all gold + pearls (BGA anti-playing)
  winner: number; // -1 none, 0/1 player, 2 draw
  winReason: number; // 0 none, 1 points, 2 crowns, 3 color, 4 anti-playing, 5 draw
  turn: number;
  ply: number;
}

export function newPlayer(): PlayerState {
  return {
    tokens: new Int8Array(NUM_TOKEN_TYPES),
    bonus: new Int8Array(6),
    colorPoints: new Int8Array(6),
    points: 0,
    crowns: 0,
    privileges: 0,
    reserved: [],
    reservedHidden: [],
    royals: [],
    royalsEarned: 0,
    numCards: 0,
  };
}

export function emptyState(): GameState {
  return {
    board: new Int8Array(BOARD_CELLS).fill(EMPTY),
    bag: new Int8Array(NUM_TOKEN_TYPES),
    players: [newPlayer(), newPlayer()],
    tablePrivileges: TOTAL_PRIVILEGES,
    pyramid: new Int8Array(PYRAMID_SLOTS).fill(EMPTY),
    decks: [[], [], [], []],
    royalsAvail: ROYALS.map((r) => r.id),
    owner: new Int8Array(NUM_CARDS).fill(-1),
    current: 0,
    firstPlayer: 0,
    phase: Phase.Main,
    replenished: false,
    lastPrivCell: -1,
    lastDiscard: 0,
    reserveTarget: -1,
    pendingCard: -1,
    effects: [],
    takeColor: -1,
    extraTurn: false,
    boughtThisTurn: false,
    passStreak: 0,
    hoard: new Int8Array(2),
    winner: -1,
    winReason: 0,
    turn: 0,
    ply: 0,
  };
}

/** Full token supply: 4 of each gem, 2 pearls, 3 gold. */
export function fullSupply(): Int8Array {
  const s = new Int8Array(NUM_TOKEN_TYPES).fill(4);
  s[PEARL] = 2;
  s[GOLD] = 3;
  return s;
}

/** Fill empty cells in spiral order with tokens drawn at random from the bag. */
export function refillBoard(s: GameState, rng: Rng): number[] {
  const drawn: number[] = [];
  for (let t = 0; t < NUM_TOKEN_TYPES; t++) for (let k = 0; k < s.bag[t]; k++) drawn.push(t);
  rng.shuffle(drawn);
  const placed: number[] = [];
  let i = 0;
  for (const cell of SPIRAL) {
    if (i >= drawn.length) break;
    if (s.board[cell] === EMPTY) {
      s.board[cell] = drawn[i++];
      placed.push(cell);
    }
  }
  s.bag.fill(0);
  return placed;
}

export function newGame(seed: number, firstPlayer?: number): GameState {
  const rng = new Rng(seed);
  const s = emptyState();
  s.bag = fullSupply();
  refillBoard(s, rng);
  for (let level = 1; level <= 3; level++) {
    const deck = rng.shuffle(LEVEL_CARD_IDS[level].slice());
    for (let k = 0; k < LEVEL_SLOT_COUNT[level]; k++) s.pyramid[LEVEL_SLOT_START[level] + k] = deck.pop()!;
    s.decks[level] = deck;
  }
  s.firstPlayer = firstPlayer ?? rng.int(2);
  s.current = s.firstPlayer;
  s.players[1 - s.firstPlayer].privileges = 1;
  s.tablePrivileges = TOTAL_PRIVILEGES - 1;
  return s;
}

function clonePlayer(p: PlayerState): PlayerState {
  return {
    tokens: p.tokens.slice(),
    bonus: p.bonus.slice(),
    colorPoints: p.colorPoints.slice(),
    points: p.points,
    crowns: p.crowns,
    privileges: p.privileges,
    reserved: p.reserved.slice(),
    reservedHidden: p.reservedHidden.slice(),
    royals: p.royals.slice(),
    royalsEarned: p.royalsEarned,
    numCards: p.numCards,
  };
}

export function cloneState(s: GameState): GameState {
  return {
    board: s.board.slice(),
    bag: s.bag.slice(),
    players: [clonePlayer(s.players[0]), clonePlayer(s.players[1])],
    tablePrivileges: s.tablePrivileges,
    pyramid: s.pyramid.slice(),
    decks: [[], s.decks[1].slice(), s.decks[2].slice(), s.decks[3].slice()],
    royalsAvail: s.royalsAvail.slice(),
    owner: s.owner.slice(),
    current: s.current,
    firstPlayer: s.firstPlayer,
    phase: s.phase,
    replenished: s.replenished,
    lastPrivCell: s.lastPrivCell,
    lastDiscard: s.lastDiscard,
    reserveTarget: s.reserveTarget,
    pendingCard: s.pendingCard,
    effects: s.effects.slice(),
    takeColor: s.takeColor,
    extraTurn: s.extraTurn,
    boughtThisTurn: s.boughtThisTurn,
    passStreak: s.passStreak,
    hoard: s.hoard.slice(),
    winner: s.winner,
    winReason: s.winReason,
    turn: s.turn,
    ply: s.ply,
  };
}

export function tokenCount(p: PlayerState): number {
  let n = 0;
  for (let t = 0; t < NUM_TOKEN_TYPES; t++) n += p.tokens[t];
  return n;
}

export function bagCount(s: GameState): number {
  let n = 0;
  for (let t = 0; t < NUM_TOKEN_TYPES; t++) n += s.bag[t];
  return n;
}
