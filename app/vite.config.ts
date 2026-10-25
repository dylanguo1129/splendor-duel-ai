import { defineConfig } from "vite";

export default defineConfig({
  root: "app",
  base: "./",
  publicDir: "public",
  build: { outDir: "../dist/app", emptyOutDir: true, target: "es2022" },
  esbuild: { target: "es2022" },
  worker: { format: "es" },
  server: { port: 5178, strictPort: true },
});
