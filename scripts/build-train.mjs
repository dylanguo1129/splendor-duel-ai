// Bundle the training loop and its worker for plain `node` (fast worker startup, no TS loader).
import { build } from "esbuild";

await build({
  entryPoints: {
    loop: "train/loop.ts",
    worker: "train/worker.ts",
    report: "train/report.ts",
    match: "train/match.ts",
    "gpu-bench": "train/gpu-bench.ts",
  },
  outdir: "dist/train",
  bundle: true,
  platform: "node",
  format: "esm",
  target: "node22",
  outExtension: { ".js": ".mjs" },
  sourcemap: true,
  logLevel: "info",
});
