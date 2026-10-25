// Compare ai/net.ts (plain JS and WebAssembly SIMD) with PyTorch on recorded positions.
//   python -I train/dump_net.py <model.json> <file.bin> 500 <dump.json>
//   npx tsx scripts/check-net-parity.ts <model.json> <dump.json>
import { readFileSync } from "node:fs";
import { Net } from "../ai/net";

const [modelPath, dumpPath] = process.argv.slice(2);
const json = JSON.parse(readFileSync(modelPath, "utf8"));
const dump = JSON.parse(readFileSync(dumpPath, "utf8")) as { entity: boolean; rows: { x: number[]; actions: number[]; logits: number[]; value: number }[] };
let failed = false;
for (const simd of [false, true]) {
  const net = new Net(json, { simd });
  if (simd && !net.usesSimd) {
    console.log("SIMD unavailable here: skipped");
    continue;
  }
  if (net.entity !== dump.entity) throw new Error(`entity mismatch: net ${net.entity}, dump ${dump.entity}`);
  let worstLogit = 0;
  let worstValue = 0;
  const logits = new Float32Array(512);
  for (const r of dump.rows) {
    const x = Float32Array.from(r.x);
    const v = net.forward(x, r.actions, r.actions.length, logits);
    worstValue = Math.max(worstValue, Math.abs(v - r.value));
    r.actions.forEach((_, k) => (worstLogit = Math.max(worstLogit, Math.abs(logits[k] - r.logits[k]))));
  }
  const ok = worstLogit < 2e-3 && worstValue < 1e-4;
  failed ||= !ok;
  console.log(`${simd ? "SIMD" : "JS  "} entity=${net.entity}: ${dump.rows.length} positions, max |dlogit| ${worstLogit.toExponential(2)}, max |dvalue| ${worstValue.toExponential(2)} ${ok ? "ok" : "MISMATCH"}`);
}
if (failed) process.exit(1);
