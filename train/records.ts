// Fixed-size training records shared with train/train.py:
//   x      : F  x uint16 (float16 bits)
//   legal  : 28 x uint8  (224-bit mask)
//   pi     : 224 x uint16 (probability * 65535)
//   z      : int8 (+1 win / -1 loss / 0 draw for the player to move)
//   q      : int8 root search value * 127 (player to move)
//   hasQ   : uint8 (1 when q is present; older records have 0)
//   flags  : uint8, bit 0 = value only (no policy target: moves of league opponents)
//   left   : uint16 actions from this position until the game ended (moves-left head target)
//   final  : 3 x int8 final prestige, crowns and best single-colour prestige of the mover minus the
//            opponent's, then uint8 GameState.winReason (auxiliary targets: how the game ended)
// F is NUM_FEATURES; each data dir records its F in meta.json, `ml: true` when its records have
// `left` and `sc: true` when they have `final` (dirs without meta.json use the base layout).
import { NUM_ACTIONS } from "../engine/constants";
import { NUM_FEATURES } from "../engine/features";
import { GameState } from "../engine/state";

export const RECORD_BYTES = 2 * NUM_FEATURES + 28 + 2 * NUM_ACTIONS + 10;

/** Final tallies of a finished game, per player, for the auxiliary targets. */
export interface FinalStats {
  points: [number, number];
  crowns: [number, number];
  bestColor: [number, number]; // highest prestige in a single colour
  reason: number; // GameState.winReason
}

export function finalStats(s: GameState): FinalStats {
  const best = (p: number) => Math.max(...Array.from(s.players[p].colorPoints.subarray(1, 6)));
  return {
    points: [s.players[0].points, s.players[1].points],
    crowns: [s.players[0].crowns, s.players[1].crowns],
    bestColor: [best(0), best(1)],
    reason: s.winReason,
  };
}

const f32 = new Float32Array(1);
const u32 = new Uint32Array(f32.buffer);
export function toHalf(v: number): number {
  f32[0] = v;
  const x = u32[0];
  const sign = (x >>> 16) & 0x8000;
  let exp = ((x >>> 23) & 0xff) - 127 + 15;
  let mant = x & 0x7fffff;
  if (exp <= 0) {
    if (exp < -10) return sign;
    mant = (mant | 0x800000) >>> (1 - exp);
    return sign | ((mant + 0x1000) >>> 13);
  }
  if (exp >= 31) return sign | 0x7c00;
  const half = sign | (exp << 10) | (mant >>> 13);
  return mant & 0x1000 ? half + 1 : half; // round half up
}

export interface Sample {
  x: Float32Array;
  legal: number[];
  pi: Map<number, number>; // action -> probability
  piPruned?: Map<number, number>; // the same with exploration visits removed (ai/mcts.ts prunedVisits)
  mover: number;
  q: number; // root search value for the mover, in [-1, 1]
  ply: number; // GameState.ply at this position (actions so far)
  /** League opponent's move: train the value on the outcome only (its search is biased or weaker). */
  valueOnly?: boolean;
}

/** z: outcome for the mover; left: actions from the sample until the game ended. */
export function writeRecord(buf: Buffer, off: number, smp: Sample, z: number, left: number, fin: FinalStats): void {
  for (let i = 0; i < NUM_FEATURES; i++) buf.writeUInt16LE(toHalf(smp.x[i]), off + 2 * i);
  let o = off + 2 * NUM_FEATURES;
  buf.fill(0, o, o + 28);
  for (const a of smp.legal) buf[o + (a >> 3)] |= 1 << (a & 7);
  o += 28;
  buf.fill(0, o, o + 2 * NUM_ACTIONS);
  for (const [a, p] of smp.pi) buf.writeUInt16LE(Math.round(p * 65535), o + 2 * a);
  o += 2 * NUM_ACTIONS;
  buf.writeInt8(z, o);
  buf.writeInt8(smp.valueOnly ? 0 : Math.max(-127, Math.min(127, Math.round(smp.q * 127))), o + 1);
  buf.writeUInt8(smp.valueOnly ? 0 : 1, o + 2);
  buf.writeUInt8(smp.valueOnly ? 1 : 0, o + 3);
  buf.writeUInt16LE(Math.max(0, Math.min(65535, left)), o + 4);
  const me = smp.mover;
  const diff = (v: [number, number]) => Math.max(-127, Math.min(127, v[me] - v[1 - me]));
  buf.writeInt8(diff(fin.points), o + 6);
  buf.writeInt8(diff(fin.crowns), o + 7);
  buf.writeInt8(diff(fin.bestColor), o + 8);
  buf.writeUInt8(fin.reason, o + 9);
}
