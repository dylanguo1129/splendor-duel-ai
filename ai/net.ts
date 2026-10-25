// Pure-JS MLP inference (Node, browser, workers). Weights are exported by train/train.py.
import { LINES } from "../engine/board";
import { A_BUY, A_CELL, A_RESERVE, NUM_ACTIONS } from "../engine/constants";
import { encode, NUM_FEATURES } from "../engine/features";
import { Rng } from "../engine/rng";
import { GameState } from "../engine/state";
import type { Evaluator } from "./mcts";
import { SimdKernels, simdKernels } from "./simd";

export interface NetJson {
  version: number;
  F: number;
  A: number;
  hidden: { in: number; out: number; W: string; b: string }[]; // W input-major [in][out]
  Wp: string; // [A][H] output-major
  bp: string;
  Wv: string; // [H]
  bv: number;
  Wm?: string; // [H] moves-left head (output * movesScale = actions until the game ends)
  bm?: number;
  movesScale?: number;
  E?: string; // [ENT_OUT][H] output-major: entity policy terms (train/entity.py)
  meta?: Record<string, unknown>; // meta.ml: the moves-left head was trained
}

// Entity policy terms (train/entity.py entity_logits): logits of moves on a card or on board cells get
// a score shared across slots, from a state query q = E h of ENT_OUT values:
// [buy card 40 | reserve card 40 | line cell by token type 7 | single cell by token type 7].
const CARD_DIM = 28 + 12;
const ENT_OUT = 2 * CARD_DIM + 14;
const ENT_PAD = Math.ceil(ENT_OUT / 4) * 4; // the SIMD layer kernel writes outputs in fours
const ME_BLOCK = 183;
const OPP_BLOCK = 213;
const PYRAMID = 243;
const MY_RESERVED = 579;

/**
 * Entity score of pyramid slot 0..11 or my reserved card 12..14 for query q[qo..qo+CARD_DIM): the card
 * block dotted with q, plus each player's per-colour cost shortfall (have: tokens + bonus, me then opponent).
 */
function cardScore(x: Float32Array, slot: number, have: Int32Array, q: Float32Array, qo: number): number {
  const c = slot < 12 ? PYRAMID + 28 * slot : MY_RESERVED + 28 * (slot - 12);
  if (!(x[c] > 0)) return 0; // empty slot: all zeros
  let z = 0;
  for (let i = 0; i < 28; i++) z += x[c + i] * q[qo + i];
  for (let t = 0; t < 6; t++) {
    const cost = Math.round(x[c + 11 + t] * 8);
    z += (Math.max(0, cost - have[t]) * q[qo + 28 + t] + Math.max(0, cost - have[6 + t]) * q[qo + 34 + t]) / 8;
  }
  return z;
}

/** Tokens + bonus per colour pearl..red for me (0..5) and the opponent (6..11), from the player blocks. */
function haves(x: Float32Array, out: Int32Array): void {
  for (let p = 0; p < 2; p++) {
    const base = p === 0 ? ME_BLOCK : OPP_BLOCK;
    for (let t = 0; t < 6; t++) out[6 * p + t] = Math.round(x[base + t] * 4) + (t > 0 ? Math.round(x[base + 7 + t] * 5) : 0);
  }
}

/** Token type 0..6 on each board cell, -1 when empty (board one-hot at x[0..175)). */
function boardTypes(x: Float32Array, out: Int8Array): void {
  for (let cell = 0; cell < 25; cell++) {
    out[cell] = -1;
    for (let t = 0; t < 7; t++) if (x[cell * 7 + t] > 0.5) out[cell] = t;
  }
}

function decodeF32(b64: string): Float32Array {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

export class Net {
  readonly F: number;
  readonly A: number;
  private layers: { in: number; out: number; W: Float32Array; b: Float32Array }[];
  private Wp: Float32Array;
  private bp: Float32Array;
  private Wv: Float32Array;
  private bv: number;
  private Wm: Float32Array | null;
  private bm: number;
  private movesScale: number;
  private E: Float32Array | null;
  private q = new Float32Array(ENT_OUT);
  private have = new Int32Array(12);
  private types = new Int8Array(25);
  private acts: Float32Array[];
  readonly H: number;
  meta?: Record<string, unknown>;
  /** Actions until the game ends, estimated at the last forward() (NaN without a trained head). */
  movesLeft = NaN;
  /** The moves-left head was trained, so the search may use it. */
  readonly hasMovesLeft: boolean;

  /** opts.simd = false: always use the plain JS loops (tests, comparisons). */
  constructor(j: NetJson, opts: { simd?: boolean } = {}) {
    // models trained before features were appended read only the first j.F inputs (same results)
    if (j.F > NUM_FEATURES || j.A !== NUM_ACTIONS || j.hidden[0]?.in !== j.F)
      throw new Error(`net shape ${j.F}x${j.A} does not fit ${NUM_FEATURES}x${NUM_ACTIONS}`);
    this.F = j.F;
    this.A = j.A;
    this.layers = j.hidden.map((l) => ({ in: l.in, out: l.out, W: decodeF32(l.W), b: decodeF32(l.b) }));
    this.H = this.layers[this.layers.length - 1].out;
    this.Wp = decodeF32(j.Wp);
    this.bp = decodeF32(j.bp);
    this.Wv = decodeF32(j.Wv);
    this.bv = j.bv;
    this.Wm = j.Wm ? decodeF32(j.Wm) : null;
    this.bm = j.bm ?? 0;
    this.movesScale = j.movesScale ?? 100;
    this.E = j.E ? decodeF32(j.E) : null;
    this.acts = this.layers.map((l) => new Float32Array(l.out));
    this.meta = j.meta;
    this.hasMovesLeft = this.Wm !== null && j.meta?.ml === true;
    this.simd = opts.simd !== false && this.layers.every((l) => l.out % 4 === 0) ? this.setUpSimd() : null;
  }

  /** The policy has entity terms (train/entity.py). */
  get entity(): boolean {
    return this.E !== null;
  }

  /** WebAssembly SIMD path (ai/simd.ts): weights copied once into wasm memory. */
  private simd: {
    k: SimdKernels;
    f32: Float32Array;
    x: number;
    W: number[];
    b: number[];
    act: number[];
    wp: number;
    wv: number;
    wm: number;
    we: number; // E transposed to input-major [H][ENT_PAD] for the layer kernel (skips zero inputs)
    zb: number; // zero bias for it
    q: number;
  } | null;
  /** Whether forward() runs on the WebAssembly SIMD kernels (false: plain JS). */
  get usesSimd(): boolean {
    return this.simd !== null;
  }

  private setUpSimd(): Net["simd"] {
    let top = 0;
    const take = (floats: number) => {
      const at = top;
      top += Math.ceil(floats / 4) * 4; // 16-byte aligned
      return at;
    };
    const x = take(NUM_FEATURES);
    const W = this.layers.map((l) => take(l.in * l.out));
    const b = this.layers.map((l) => take(l.out));
    const act = this.layers.map((l) => take(l.out));
    const wp = take(this.A * this.H);
    const wv = take(this.H);
    const wm = take(this.H);
    const we = this.E ? take(this.H * ENT_PAD) : 0;
    const zb = this.E ? take(ENT_PAD) : 0;
    const q = this.E ? take(ENT_PAD) : 0;
    let memory: WebAssembly.Memory;
    try {
      memory = new WebAssembly.Memory({ initial: Math.ceil((top * 4) / 65536) + 1 });
    } catch {
      return null;
    }
    const k = simdKernels(memory);
    if (!k) return null;
    const f32 = new Float32Array(memory.buffer);
    this.layers.forEach((l, i) => {
      f32.set(l.W, W[i]);
      f32.set(l.b, b[i]);
    });
    f32.set(this.Wp, wp);
    f32.set(this.Wv, wv);
    if (this.Wm) f32.set(this.Wm, wm);
    if (this.E) {
      const H = this.H;
      for (let j = 0; j < ENT_OUT; j++) for (let i = 0; i < H; i++) f32[we + i * ENT_PAD + j] = this.E[j * H + i];
    }
    return { k, f32, x, W, b, act, wp, wv, wm, we, zb, q };
  }

  /** Adds the entity terms (from this.q, already computed) to the logits of the given actions. */
  private addEntity(x: Float32Array, actions: Int16Array | number[], n: number, logits: Float32Array): void {
    const q = this.q;
    let haveDone = false;
    let typesDone = false;
    for (let k = 0; k < n; k++) {
      const a = actions[k];
      if (a < A_CELL + 25) {
        if (!typesDone) {
          boardTypes(x, this.types);
          typesDone = true;
        }
        if (a < A_CELL) {
          for (const cell of LINES[a]) if (this.types[cell] >= 0) logits[k] += q[2 * CARD_DIM + this.types[cell]];
        } else if (this.types[a - A_CELL] >= 0) logits[k] += q[2 * CARD_DIM + 7 + this.types[a - A_CELL]];
        continue;
      }
      const reserve = a >= A_RESERVE && a < A_RESERVE + 12;
      const buy = a >= A_BUY && a < A_BUY + 15;
      if (!reserve && !buy) continue;
      if (!haveDone) {
        haves(x, this.have);
        haveDone = true;
      }
      logits[k] += reserve ? cardScore(x, a - A_RESERVE, this.have, q, CARD_DIM) : cardScore(x, a - A_BUY, this.have, q, 0);
    }
  }

  /** Returns value in [-1,1]; writes logits for the given actions into `logits` (same order). */
  forward(x: Float32Array, actions: Int16Array | number[], n: number, logits: Float32Array): number {
    const s = this.simd;
    if (s) {
      const F0 = this.layers[0].in;
      s.f32.set(x.length === F0 ? x : x.subarray(0, F0), s.x);
      let inPtr = s.x * 4;
      for (let li = 0; li < this.layers.length; li++) {
        const L = this.layers[li];
        s.k.layer(inPtr, L.in, s.W[li] * 4, s.b[li] * 4, s.act[li] * 4, L.out, 1);
        inPtr = s.act[li] * 4;
      }
      const H = this.H;
      for (let k = 0; k < n; k++) {
        const a = actions[k];
        logits[k] = this.bp[a] + s.k.dot(inPtr, (s.wp + a * H) * 4, H);
      }
      if (this.E) {
        s.k.layer(inPtr, H, s.we * 4, s.zb * 4, s.q * 4, ENT_PAD, 0);
        this.q.set(s.f32.subarray(s.q, s.q + ENT_OUT));
        this.addEntity(x, actions, n, logits);
      }
      if (this.hasMovesLeft) this.movesLeft = Math.max(0, (this.bm + s.k.dot(inPtr, s.wm * 4, H)) * this.movesScale);
      return Math.tanh(this.bv + s.k.dot(inPtr, s.wv * 4, H));
    }
    let input = x;
    for (let li = 0; li < this.layers.length; li++) {
      const L = this.layers[li];
      const h = this.acts[li];
      h.set(L.b);
      const W = L.W;
      const out = L.out;
      for (let j = 0; j < L.in; j++) {
        const xj = input[j];
        if (xj === 0) continue;
        const base = j * out;
        for (let i = 0; i < out; i++) h[i] += xj * W[base + i];
      }
      for (let i = 0; i < out; i++) if (h[i] < 0) h[i] = 0;
      input = h;
    }
    const H = this.H;
    for (let k = 0; k < n; k++) {
      const a = actions[k];
      const base = a * H;
      let z = this.bp[a];
      for (let i = 0; i < H; i++) z += input[i] * this.Wp[base + i];
      logits[k] = z;
    }
    if (this.E) {
      for (let j = 0; j < ENT_OUT; j++) {
        let z = 0;
        for (let i = 0; i < H; i++) z += input[i] * this.E[j * H + i];
        this.q[j] = z;
      }
      this.addEntity(x, actions, n, logits);
    }
    let v = this.bv;
    for (let i = 0; i < H; i++) v += input[i] * this.Wv[i];
    if (this.hasMovesLeft) {
      let m = this.bm;
      for (let i = 0; i < H; i++) m += input[i] * this.Wm![i];
      this.movesLeft = Math.max(0, m * this.movesScale);
    }
    return Math.tanh(v);
  }
}

export class NetEvaluator implements Evaluator {
  private x = new Float32Array(NUM_FEATURES);
  private logits = new Float32Array(512);
  movesLeft?: number;
  constructor(
    private net: Net,
    private temperature = 1,
  ) {}
  evaluate(s: GameState, legal: Int16Array, n: number, priors: Float32Array, _rng: Rng): number {
    encode(s, this.x);
    const v = this.net.forward(this.x, legal, n, this.logits);
    if (this.net.hasMovesLeft) this.movesLeft = this.net.movesLeft;
    let max = -Infinity;
    for (let i = 0; i < n; i++) if (this.logits[i] > max) max = this.logits[i];
    for (let i = 0; i < n; i++) priors[i] = Math.exp((this.logits[i] - max) / this.temperature);
    return v;
  }
}
