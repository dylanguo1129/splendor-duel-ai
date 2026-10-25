// Token types. Gem codes 0..5 match BGA; gold is BGA color -1 / type 1.
export const PEARL = 0;
export const BLUE = 1;
export const WHITE = 2;
export const GREEN = 3;
export const BLACK = 4;
export const RED = 5;
export const GOLD = 6;
export const NUM_TOKEN_TYPES = 7;
export const GEM_COLORS = [BLUE, WHITE, GREEN, BLACK, RED] as const;
export const TOKEN_NAMES = ["珍珠", "蓝", "白", "绿", "黑", "红", "金"];
export const TOKEN_NAMES_EN = ["pearl", "blue", "white", "green", "black", "red", "gold"];

// Card colors: 1..5 gem colors, 9 = joker / colorless.
export const COLOR_JOKER = 9;

export const EMPTY = -1;
export const BOARD_CELLS = 25;
export const TOTAL_TOKENS = 25;
export const TOTAL_PRIVILEGES = 3;
export const MAX_TOKENS = 10;
export const MAX_RESERVED = 3;

// Pyramid slots: L1 0..4, L2 5..8, L3 9..11.
export const PYRAMID_SLOTS = 12;
export const LEVEL_SLOT_START = [0, 0, 5, 9]; // indexed by level
export const LEVEL_SLOT_COUNT = [0, 5, 4, 3];

export const WIN_POINTS = 20;
export const WIN_CROWNS = 10;
export const WIN_COLOR_POINTS = 10;
export const ROYAL_THRESHOLDS = [3, 6];

export const enum Phase {
  Main = 0,
  ReserveGold = 1,
  Joker = 2,
  TakeToken = 3,
  Steal = 4,
  Royal = 5,
  Discard = 6,
  GameOver = 7,
}

// ---- Action ids (also the NN policy index) ----
export const A_TAKE_LINE = 0; // + line index (0..144)
export const NUM_LINES = 145;
export const A_CELL = 145; // + cell (0..24)
export const A_REPLENISH = 170;
export const A_RESERVE = 171; // + target (0..11 pyramid slot, 12..14 deck L1..L3)
export const A_BUY = 186; // + source (0..11 pyramid slot, 12..14 reserved slot)
export const A_JOKER = 201; // + (color - 1), color 1..5
export const A_STEAL = 206; // + token type 0..5
export const A_ROYAL = 212; // + royal id 0..3
export const A_DISCARD = 216; // + token type 0..6
export const A_PASS = 223;
export const NUM_ACTIONS = 224;

export const DRAW = 2; // winner value for a draw
export const MAX_TURNS = 150; // safety cap (BGA has none); reaching it is a draw
