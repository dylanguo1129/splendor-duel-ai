// Main-thread side of parallel search: N workers search independent trees, results are merged.
import { listActions } from "../engine/rules";
import type { GameState } from "../engine/state";
import { mergeResults, SearchResult } from "./mcts";
import type { NetJson } from "./net";
import type { PlayoutRun } from "./outlook";
import type { WorkerRequest, WorkerResponse } from "./search-worker";
import { AutoTimeOptions, autoTimeCheck, StopReason, TimeDecision } from "./time-control";

export interface SearchTiming {
  spentMs: number; // search time actually used
  targetMs: number; // what auto mode planned for this position
  reason: StopReason | "forced";
}

type Extra = { rootActions?: number[]; resampleOwnHidden?: boolean };

export class SearchPool {
  private workers: Worker[] = [];
  private ready: Promise<void>;
  private nextId = 1;
  private pending = new Map<number, (m: WorkerResponse) => void>();
  private seed = 1;

  constructor(makeWorker: () => Worker, size: number, net: NetJson | null) {
    const readies: Promise<void>[] = [];
    for (let i = 0; i < size; i++) {
      const w = makeWorker();
      readies.push(
        new Promise((resolve) => {
          w.onmessage = (e: MessageEvent<WorkerResponse>) => {
            const m = e.data;
            if (m.type === "ready") resolve();
            else this.pending.get(m.id)?.(m);
          };
        }),
      );
      w.postMessage({ type: "init", net } satisfies WorkerRequest);
      this.workers.push(w);
    }
    this.ready = Promise.all(readies).then(() => undefined);
  }

  get size(): number {
    return this.workers.length;
  }

  private request(w: Worker, req: WorkerRequest & { id: number }): Promise<WorkerResponse> {
    return new Promise((resolve) => {
      this.pending.set(req.id, (m) => {
        this.pending.delete(req.id);
        resolve(m);
      });
      w.postMessage(req);
    });
  }

  private rootLegal(state: GameState, rootActions?: number[]): number[] {
    const legal = listActions(state);
    const allowed = rootActions ? legal.filter((a) => rootActions.includes(a)) : [];
    return allowed.length ? allowed : legal;
  }

  /** Search `state` for the player to move. Single legal actions return immediately. */
  async search(state: GameState, budget: { timeMs?: number; iterations?: number }, extra: Extra = {}): Promise<SearchResult> {
    const legal = this.rootLegal(state, extra.rootActions);
    if (legal.length === 1) return forced(legal[0]);
    await this.ready;
    const per = budget.iterations ? Math.ceil(budget.iterations / this.workers.length) : undefined;
    const results = await Promise.all(
      this.workers.map((w) => {
        const id = this.nextId++;
        return this.request(w, { type: "search", id, state, timeMs: budget.timeMs, iterations: per, seed: this.nextSeed(), ...extra });
      }),
    );
    return mergeResults(results.map(resultOf));
  }

  /**
   * Auto mode: search in short rounds and let the time controller decide after each round whether
   * the move is settled (see time-control.ts). `auto.base` is in milliseconds.
   */
  async searchAuto(
    state: GameState,
    auto: AutoTimeOptions,
    extra: Extra = {},
    onProgress?: (r: SearchResult, spentMs: number, d: TimeDecision) => void,
  ): Promise<SearchResult & { timing: SearchTiming }> {
    const legal = this.rootLegal(state, extra.rootActions);
    if (legal.length === 1) return { ...forced(legal[0]), timing: { spentMs: 0, targetMs: 0, reason: "forced" } };
    await this.ready;
    const round = Math.min(400, Math.max(100, auto.base / 15));
    const ids = this.workers.map(() => this.nextId++);
    try {
      let results = await Promise.all(
        this.workers.map((w, i) => this.request(w, { type: "search", id: ids[i], state, timeMs: round, seed: this.nextSeed(), keep: true, ...extra })),
      );
      // time is counted in completed rounds, so waiting for a busy worker does not eat the budget
      let spent = round;
      for (;;) {
        const merged = mergeResults(results.map(resultOf));
        const d = autoTimeCheck(auto, { spent, iterations: merged.iterations, legal: legal.length, children: merged.children });
        onProgress?.(merged, spent, d);
        if (d.stop) return { ...merged, timing: { spentMs: spent, targetMs: d.target, reason: d.reason! } };
        results = await Promise.all(this.workers.map((w, i) => this.request(w, { type: "continue", id: ids[i], timeMs: round })));
        spent += round;
      }
    } finally {
      this.workers.forEach((w, i) => w.postMessage({ type: "end", id: ids[i] } satisfies WorkerRequest));
    }
  }

  /** Policy playouts from `state` (see outlook.ts), split across the workers. */
  async playouts(state: GameState, count: number, timeMs: number, extra: Extra = {}): Promise<PlayoutRun[]> {
    await this.ready;
    const per = Math.ceil(count / this.workers.length);
    const out = await Promise.all(
      this.workers.map((w) => {
        const id = this.nextId++;
        return this.request(w, { type: "playouts", id, state, count: per, timeMs, seed: this.nextSeed(), ...extra });
      }),
    );
    return out.flatMap((m) => (m.type === "playouts" ? m.runs : []));
  }

  private nextSeed(): number {
    return (this.seed++ * 2654435761) >>> 0;
  }

  terminate(): void {
    for (const w of this.workers) w.terminate();
    this.workers = [];
  }
}

function forced(action: number): SearchResult {
  return { action, value: NaN, iterations: 0, children: [{ action, key: action, N: 1, Q: NaN, prior: 1 }], pv: [action] };
}

function resultOf(m: WorkerResponse): SearchResult {
  if (m.type !== "result") throw new Error(`unexpected worker reply ${m.type}`);
  return m.result;
}
