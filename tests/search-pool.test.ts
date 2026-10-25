// SearchPool against the real worker code, run in-process behind a fake Worker.
import { beforeAll, describe, expect, it } from "vitest";
import type { SearchPool as Pool } from "../ai/search-pool";
import { applyAction, listActions } from "../engine/rules";
import { Rng } from "../engine/rng";
import { cloneState, newGame } from "../engine/state";
import { Phase } from "../engine/constants";

let SearchPool: typeof Pool;
let sender: FakeWorker | null = null;
let handler: ((e: { data: unknown }) => void) | null = null;

class FakeWorker {
  onmessage: ((e: { data: unknown }) => void) | null = null;
  postMessage(data: unknown): void {
    const copy = structuredClone(data);
    setTimeout(() => {
      sender = this;
      handler!({ data: copy });
    }, 0);
  }
  terminate(): void {}
}

beforeAll(async () => {
  const g = globalThis as any;
  g.self = {
    postMessage: (m: unknown) => {
      const to = sender!;
      const copy = structuredClone(m);
      setTimeout(() => to.onmessage?.({ data: copy }), 0);
    },
  };
  await import("../ai/search-worker");
  handler = g.self.onmessage;
  SearchPool = (await import("../ai/search-pool")).SearchPool;
});

describe("search pool", () => {
  it("auto search stays within its budget and reports why it stopped", async () => {
    const pool = new SearchPool(() => new FakeWorker() as unknown as Worker, 2, null);
    const s = newGame(21);
    const progress: number[] = [];
    const r = await pool.searchAuto(s, { base: 1000 }, {}, (_m, spent) => progress.push(spent));
    expect(listActions(s)).toContain(r.action);
    expect(r.iterations).toBeGreaterThan(0);
    expect(r.timing.spentMs).toBeGreaterThanOrEqual(200);
    expect(r.timing.spentMs).toBeLessThanOrEqual(3000);
    expect(["decided", "target", "extended", "max"]).toContain(r.timing.reason);
    expect(progress.length).toBeGreaterThanOrEqual(2);
    expect(progress[progress.length - 1]).toBe(r.timing.spentMs);
    // a fixed search and playouts still work on the same workers afterwards
    const f = await pool.search(s, { iterations: 64 });
    expect(f.iterations).toBeGreaterThanOrEqual(64);
    const runs = await pool.playouts(s, 4, 2000);
    expect(runs).toHaveLength(4);
  }, 20000);

  it("returns forced moves immediately", async () => {
    const pool = new SearchPool(() => new FakeWorker() as unknown as Worker, 1, null);
    // advance a game until some decision has a single legal action
    const s = newGame(5);
    const rng = new Rng(9);
    let forced = null;
    for (let i = 0; i < 2000 && s.phase !== Phase.GameOver; i++) {
      const acts = listActions(s);
      if (acts.length === 1) {
        forced = cloneState(s);
        break;
      }
      applyAction(s, acts[Math.floor(rng.next() * acts.length)], rng);
    }
    expect(forced).not.toBeNull();
    const r = await pool.searchAuto(forced!, { base: 1000 });
    expect(r.timing).toMatchObject({ spentMs: 0, reason: "forced" });
    expect(r.iterations).toBe(0);
  });
});
