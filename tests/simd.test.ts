import { existsSync, readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { Net, NetJson } from "../ai/net";
import { Phase } from "../engine/constants";
import { encode, NUM_FEATURES } from "../engine/features";
import { Rng } from "../engine/rng";
import { applyAction, legalActions } from "../engine/rules";
import { newGame } from "../engine/state";

const b64 = (a: Float32Array) => Buffer.from(a.buffer, a.byteOffset, a.byteLength).toString("base64");

/** Small random model in the export format: F inputs, hidden sizes, optional moves-left head. */
function randomNet(F: number, hidden: number[], seed: number): NetJson {
  const rng = new Rng(seed);
  const rand = (n: number, scale: number) => Float32Array.from({ length: n }, () => (rng.next() - 0.5) * scale);
  const dims = [F, ...hidden];
  const H = hidden[hidden.length - 1];
  return {
    version: 1,
    F,
    A: 224,
    hidden: hidden.map((out, k) => ({ in: dims[k], out, W: b64(rand(dims[k] * out, 0.2)), b: b64(rand(out, 0.2)) })),
    Wp: b64(rand(224 * H, 0.2)),
    bp: b64(rand(224, 0.2)),
    Wv: b64(rand(H, 0.2)),
    bv: 0.05,
    Wm: b64(rand(H, 0.02)),
    bm: 0.4,
    movesScale: 100,
    meta: { ml: true },
  };
}

function positions(count: number) {
  const out: { x: Float32Array; legal: Int16Array }[] = [];
  const rng = new Rng(4);
  const buf = new Int16Array(512);
  for (let g = 0; out.length < count; g++) {
    const s = newGame(900 + g);
    for (let ply = 0; ply < 90 && s.phase !== Phase.GameOver && out.length < count; ply++) {
      const n = legalActions(s, buf);
      const x = new Float32Array(NUM_FEATURES);
      encode(s, x);
      out.push({ x, legal: buf.slice(0, n) });
      applyAction(s, buf[Math.floor(rng.next() * n)], rng);
    }
  }
  return out;
}

function compare(j: NetJson) {
  const fast = new Net(j);
  const slow = new Net(j, { simd: false });
  expect(fast.usesSimd).toBe(true);
  expect(slow.usesSimd).toBe(false);
  const la = new Float32Array(512);
  const lb = new Float32Array(512);
  let worst = 0;
  for (const { x, legal } of positions(300)) {
    const va = fast.forward(x, legal, legal.length, la);
    const vb = slow.forward(x, legal, legal.length, lb);
    worst = Math.max(worst, Math.abs(va - vb));
    if (fast.hasMovesLeft) worst = Math.max(worst, Math.abs(fast.movesLeft - slow.movesLeft) / 100);
    for (let k = 0; k < legal.length; k++) worst = Math.max(worst, Math.abs(la[k] - lb[k]));
  }
  return worst;
}

describe("WebAssembly SIMD inference", () => {
  it("matches the JS loops (2 and 3 hidden layers, base and full input layout)", () => {
    expect(compare(randomNet(NUM_FEATURES, [64, 32], 1))).toBeLessThan(1e-4);
    expect(compare(randomNet(NUM_FEATURES, [128, 128, 128], 2))).toBeLessThan(1e-4);
    expect(compare(randomNet(820, [64, 64], 3))).toBeLessThan(1e-4); // older 820-input models
  });

  it.skipIf(!existsSync("models/best.json"))("matches the JS loops on the current best model", () => {
    expect(compare(JSON.parse(readFileSync("models/best.json", "utf8")) as NetJson)).toBeLessThan(1e-3);
  });

  it("falls back to JS when a layer width is not a multiple of 4", () => {
    expect(new Net(randomNet(NUM_FEATURES, [30], 5)).usesSimd).toBe(false);
  });
});
