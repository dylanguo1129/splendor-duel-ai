// Print JS Net outputs (value, logits of actions 0..4) for the first K records; compare with train/check_net.py.
import { readFileSync } from "node:fs";
import { Net } from "../ai/net";
import { NUM_FEATURES } from "../engine/features";
import { RECORD_BYTES } from "../train/records";

const [jsonPath, binPath, kArg] = process.argv.slice(2);
const net = new Net(JSON.parse(readFileSync(jsonPath, "utf8")));
const buf = readFileSync(binPath);
const k = Number(kArg);

function half(h: number): number {
  const s = h & 0x8000 ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const m = h & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}

const x = new Float32Array(NUM_FEATURES);
const logits = new Float32Array(5);
for (let i = 0; i < k; i++) {
  const off = i * RECORD_BYTES;
  for (let j = 0; j < NUM_FEATURES; j++) x[j] = half(buf.readUInt16LE(off + 2 * j));
  const v = net.forward(x, [0, 1, 2, 3, 4], 5, logits);
  console.log(`${v.toFixed(6)} ${[...logits].map((z) => z.toFixed(6)).join(" ")}`);
}
