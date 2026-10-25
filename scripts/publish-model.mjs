// Publish a model to the local app (app/public/models/best.json), which reads it on every page load.
// Writes atomically so a page loading at the same moment never sees a half-written file.
import { copyFileSync, existsSync, mkdirSync, renameSync } from "node:fs";
import { dirname } from "node:path";

export function publishModel(src = "models/best.json") {
  if (!existsSync(src)) return [];
  const done = [];
  for (const dst of ["app/public/models/best.json"]) {
    mkdirSync(dirname(dst), { recursive: true });
    const tmp = `${dst}.tmp`;
    copyFileSync(src, tmp);
    renameSync(tmp, dst);
    done.push(dst);
  }
  return done;
}

if (import.meta.url === `file:///${process.argv[1].replace(/\\/g, "/")}`) {
  console.log("published:", publishModel(process.argv[2]).join(", ") || "nothing (no model)");
}
