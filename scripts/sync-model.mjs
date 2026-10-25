// Copy the current best network into the local app's static folder (if one exists).
import { copyFileSync, existsSync, mkdirSync, rmSync } from "node:fs";

const src = process.argv[2] ?? "models/best.json";
mkdirSync("app/public/models", { recursive: true });
if (existsSync(src)) {
  copyFileSync(src, "app/public/models/best.json");
  console.log(`app model <- ${src}`);
} else {
  rmSync("app/public/models/best.json", { force: true });
  console.log("no trained model yet: the app uses the heuristic engine");
}
