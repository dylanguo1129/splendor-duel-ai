// Single-observer, open-loop Information Set MCTS with PUCT selection.
// Every iteration samples a fresh determinization (deck order, opponent's blind reserves)
// and fresh chance outcomes (board refills), so the tree never "sees" hidden information.
import { A_BUY, A_RESERVE, DRAW, Phase } from "../engine/constants";
import { determinize } from "../engine/determinize";
import { Rng } from "../engine/rng";
import { applyAction, DEFAULT_RULES, legalActions, RulesConfig } from "../engine/rules";
import { GameState } from "../engine/state";

export interface Evaluator {
  /**
   * Value of `s` for s.current in [-1, 1], and prior weights (>= 0, need not sum to 1)
   * for the n legal actions in `legal`, written to `priors`.
   */
  evaluate(s: GameState, legal: Int16Array, n: number, priors: Float32Array, rng: Rng): number;
  /** Estimated actions until the game ends at the last evaluated state (network moves-left head). */
  movesLeft?: number;
}

/**
 * Moves-left preference ("win quickly, lose slowly"): when the player choosing is clearly winning
 * (Q > q0), children whose lines end sooner get up to `weight` extra score, when clearly losing
 * (Q < -q0) children that last longer do. `norm` actions of difference give the full effect.
 * Measured (gen47 + trained head vs gen47, 200 sims, 400 games): weight 0.2 / q0 0.2 scored 50.2%
 * and converted won positions 16% faster (35.6 vs 42.2 actions); 0.04 / 0.5 had no visible effect.
 */
export interface MovesLeftOptions {
  weight: number;
  norm: number;
  q0: number;
}
export const DEFAULT_MLH: MovesLeftOptions = { weight: 0.2, norm: 20, q0: 0.2 };
/** First-play urgency relative to the parent used with a network (gen51_2 at 200 sims: 60% vs the fixed fpu). */
export const DEFAULT_FPU_REDUCTION = 0.3;

export interface MctsOptions {
  iterations?: number;
  timeMs?: number;
  cPuct?: number;
  fpu?: number; // value assumed for unvisited children (mover's view)
  /** Unvisited children: the parent's current value minus this, instead of the fixed `fpu`. */
  fpuReduction?: number;
  /** The same at the root only (default: `fpuReduction`), e.g. smaller so that root noise still explores. */
  rootFpuReduction?: number;
  /** Scale the reduction by sqrt(prior mass of the children visited so far), as KataGo and lc0 do. */
  fpuScaled?: boolean;
  dirichletAlpha?: number; // root noise (self-play only)
  dirichletWeight?: number;
  seed?: number;
  rules?: RulesConfig;
  /** Only these actions may be chosen at the root (e.g. part of a multi-step move was already played). */
  rootActions?: number[];
  /** Spectator view: the root player's own blind reservations are unknown too. */
  resampleOwnHidden?: boolean;
  /** Prefer quicker wins / slower losses with the evaluator's moves-left estimate (off: undefined). */
  mlh?: MovesLeftOptions;
}

export class Node {
  N = 0;
  W = 0; // sum of values from `mover`'s perspective
  M = 0; // sum of estimated actions left from this node's state (one per visit, like W)
  /** Visited children only (created on first selection), keyed by actionKey. */
  children: Map<number, Node> | null = null;
  /** Prior per action key, set when the node is expanded. */
  priors: Map<number, number> | null = null;
  constructor(
    public action: number,
    public mover: number,
    public prior: number,
  ) {}
  get expanded(): boolean {
    return this.priors !== null;
  }
}

/** A leaf awaiting evaluation (see Mcts.beginIteration). */
export interface PendingLeaf {
  s: GameState; // determinized state at the leaf
  node: Node;
  path: Node[];
  legal: Int16Array; // legal actions at the leaf (length n)
  n: number;
  chooser: number; // player to move at the leaf
}

export interface ChildStat {
  action: number;
  key: number;
  N: number;
  Q: number; // mover's (= root player's) mean value
  prior: number;
  M?: number; // mean estimated actions until the game ends after this move (moves-left search)
}

export interface SearchResult {
  action: number;
  value: number; // root player's expected value in [-1, 1]
  iterations: number;
  children: ChildStat[]; // sorted by visits, desc
  pv: number[]; // principal variation (most-visited path)
}

/**
 * Policy-target pruning (KataGo, arXiv 1902.10565 sec. 3.1): visits that only exploration produced
 * (root noise, first-play urgency) are removed from the policy target. Every child except the most
 * visited keeps only as many visits as PUCT would have given it under the un-noised `prior` at its
 * final value; children left with a single visit are dropped. Returns kept visits per child key.
 */
export function prunedVisits(children: ChildStat[], prior: Map<number, number>, cPuct: number): Map<number, number> {
  const kept = new Map<number, number>();
  let total = 0;
  let best: ChildStat | null = null;
  for (const c of children) {
    total += c.N;
    if (!best || c.N > best.N) best = c;
  }
  if (!best || best.N === 0) return kept;
  const sqrtN = Math.sqrt(total + 1);
  const p = (c: ChildStat) => prior.get(c.key) ?? c.prior;
  const bestScore = best.Q + (cPuct * p(best) * sqrtN) / (1 + best.N);
  for (const c of children) {
    if (c.N === 0) continue;
    if (c === best || c.Q >= bestScore) {
      kept.set(c.key, c.N);
      continue;
    }
    // fewest visits at which this child's PUCT score no longer exceeds the best child's
    const n = Math.min(c.N, Math.max(0, Math.ceil((cPuct * p(c) * sqrtN) / (bestScore - c.Q) - 1)));
    if (n > 1) kept.set(c.key, n);
  }
  return kept;
}

/** Child key: buy/reserve of a pyramid slot also identifies the card (slot contents vary by determinization). */
export function actionKey(s: GameState, a: number): number {
  if ((a >= A_BUY && a < A_BUY + 12) || (a >= A_RESERVE && a < A_RESERVE + 12)) {
    const slot = a >= A_BUY ? a - A_BUY : a - A_RESERVE;
    return a + 1024 * (s.pyramid[slot] + 1);
  }
  return a;
}

function terminalValue(s: GameState, player: number): number {
  return s.winner === DRAW ? 0 : s.winner === player ? 1 : -1;
}

function sampleGamma(alpha: number, rng: Rng): number {
  // Marsaglia-Tsang for alpha >= 1, boost for alpha < 1
  if (alpha < 1) return sampleGamma(alpha + 1, rng) * Math.pow(rng.next() || 1e-12, 1 / alpha);
  const d = alpha - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x: number;
    let v: number;
    do {
      const u1 = rng.next() || 1e-12;
      const u2 = rng.next();
      x = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      v = 1 + c * x;
    } while (v <= 0);
    v = v * v * v;
    const u = rng.next();
    if (u < 1 - 0.0331 * x * x * x * x || Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}

export class Mcts {
  root: Node;
  private rng: Rng;
  private legal = new Int16Array(512);
  private keys = new Int32Array(512);
  private edgePrior = new Float32Array(512);
  private priors = new Float32Array(512);
  private opts: Required<Omit<MctsOptions, "iterations" | "timeMs" | "seed" | "rules" | "rootActions" | "resampleOwnHidden" | "mlh" | "fpuReduction" | "rootFpuReduction" | "fpuScaled">> & {
    rules: RulesConfig;
  };
  private rootAllowed: Set<number> | null;
  private resampleOwn: boolean;
  private mlh: MovesLeftOptions | null;
  private fpuReduction: number | null;
  private rootFpuReduction: number | null;
  private fpuScaled: boolean;
  /** Root priors by child key before Dirichlet noise, normalized (set when the root is expanded). */
  rootPriors: Map<number, number> | null = null;

  constructor(
    private rootState: GameState,
    private evaluator: Evaluator,
    opts: MctsOptions = {},
  ) {
    this.rng = new Rng(opts.seed ?? 12345);
    this.opts = {
      cPuct: opts.cPuct ?? 1.5,
      fpu: opts.fpu ?? -0.2,
      dirichletAlpha: opts.dirichletAlpha ?? 0,
      dirichletWeight: opts.dirichletWeight ?? 0.25,
      rules: opts.rules ?? DEFAULT_RULES,
    };
    this.rootAllowed = opts.rootActions ? new Set(opts.rootActions) : null;
    this.resampleOwn = opts.resampleOwnHidden ?? false;
    this.mlh = opts.mlh ?? null;
    this.fpuReduction = opts.fpuReduction ?? null;
    this.rootFpuReduction = opts.rootFpuReduction ?? this.fpuReduction;
    this.fpuScaled = opts.fpuScaled ?? false;
    this.root = new Node(-1, 1 - rootState.current, 1);
  }

  /** Legal actions at `s`, restricted to the allowed root actions when at the root. */
  private legalAt(s: GameState, atRoot: boolean): number {
    let n = legalActions(s, this.legal);
    if (atRoot && this.rootAllowed) {
      let k = 0;
      for (let i = 0; i < n; i++) if (this.rootAllowed.has(this.legal[i])) this.legal[k++] = this.legal[i];
      if (k > 0) n = k;
    }
    return n;
  }

  run(opts: { iterations?: number; timeMs?: number }): number {
    const deadline = opts.timeMs ? performance.now() + opts.timeMs : Infinity;
    const maxIt = opts.iterations ?? Infinity;
    let it = 0;
    while (it < maxIt) {
      this.iterate();
      it++;
      if ((it & 15) === 0 && performance.now() >= deadline) break;
    }
    return it;
  }

  private iterate(): void {
    const p = this.beginIteration();
    if (!p) return;
    const v = this.evaluator.evaluate(p.s, p.legal, p.n, this.priors, this.rng);
    this.completeIteration(p, v, this.priors, this.evaluator.movesLeft ?? 0);
  }

  /**
   * Run selection for one iteration. Returns null when the iteration already finished (terminal
   * leaf, backed up), or the leaf that needs an evaluation: call completeIteration with its value
   * (for p.chooser) and priors in p.legal order. Allows batching evaluations across many trees.
   */
  beginIteration(): PendingLeaf | null {
    const me = this.rootState.current;
    const s = determinize(this.rootState, me, this.rng, this.resampleOwn);
    const path: Node[] = [this.root];
    let node = this.root;
    for (;;) {
      if (s.phase === Phase.GameOver) {
        this.backup(path, terminalValue(s, 0), 0);
        return null;
      }
      const n = this.legalAt(s, node === this.root);
      const chooser = s.current;
      if (!node.expanded) return { s, node, path, legal: this.legal.slice(0, n), n, chooser };
      // select among currently legal actions (PUCT); child nodes exist only once visited
      const pri = node.priors!;
      const kids = node.children!;
      let sumPrior = 0;
      let sumN = 0;
      let visitedPrior = 0;
      for (let i = 0; i < n; i++) {
        const key = actionKey(s, this.legal[i]);
        this.keys[i] = key;
        let pr = pri.get(key);
        if (pr === undefined) {
          // action that was not legal in earlier determinizations: average prior
          pr = 1 / n;
          pri.set(key, pr);
        }
        this.edgePrior[i] = pr;
        sumPrior += pr;
        const kn = kids.get(key)?.N ?? 0;
        sumN += kn;
        if (kn > 0) visitedPrior += pr;
      }
      const sqrtN = Math.sqrt(sumN + 1);
      const cp = this.opts.cPuct / (sumPrior || 1);
      const mlh = this.mlh;
      const mNode = node.N > 0 ? node.M / node.N : 0;
      let fpu = this.opts.fpu;
      const reduction = node === this.root ? this.rootFpuReduction : this.fpuReduction;
      if (reduction !== null && node.N > 0) {
        // the root keeps W for the player to move there; other nodes for the player who moved into them
        const parentQ = node === this.root || node.mover === chooser ? node.W / node.N : -node.W / node.N;
        // KataGo / lc0 style: the reduction grows with the policy mass already visited
        fpu = parentQ - (this.fpuScaled ? reduction * Math.sqrt(visitedPrior / (sumPrior || 1)) : reduction);
      }
      let bestI = 0;
      let bestScore = -Infinity;
      for (let i = 0; i < n; i++) {
        const ch = kids.get(this.keys[i]);
        const cn = ch ? ch.N : 0;
        const q = cn > 0 ? ch!.W / cn : fpu;
        let score = q + (cp * this.edgePrior[i] * sqrtN) / (1 + cn);
        if (mlh && cn > 0) {
          // winning: prefer lines that end sooner than this node's average; losing: later
          const sure = q > mlh.q0 ? (q - mlh.q0) / (1 - mlh.q0) : q < -mlh.q0 ? (q + mlh.q0) / (1 - mlh.q0) : 0;
          if (sure !== 0) {
            const d = (mNode - ch!.M / cn) / mlh.norm;
            score += mlh.weight * sure * (d > 1 ? 1 : d < -1 ? -1 : d);
          }
        }
        if (score > bestScore) {
          bestScore = score;
          bestI = i;
        }
      }
      let best = kids.get(this.keys[bestI]);
      if (!best) {
        best = new Node(this.legal[bestI], chooser, this.edgePrior[bestI]);
        kids.set(this.keys[bestI], best);
      }
      applyAction(s, best.action, this.rng, this.opts.rules);
      node = best;
      path.push(node);
    }
  }

  /**
   * Expand the pending leaf with its evaluation and back the value up the path. `movesLeft`: the
   * evaluator's estimate of actions until the game ends at the leaf (0 when it has none).
   */
  completeIteration(p: PendingLeaf, v: number, priors: Float32Array, movesLeft = 0): void {
    const { node, s, n, chooser } = p;
    if (node === this.root) {
      let sum = 0;
      for (let i = 0; i < n; i++) sum += priors[i];
      this.rootPriors = new Map();
      for (let i = 0; i < n; i++) this.rootPriors.set(actionKey(s, p.legal[i]), priors[i] / (sum || 1));
      if (this.opts.dirichletAlpha > 0) this.addNoise(priors, n);
    }
    const pri = new Map<number, number>();
    for (let i = 0; i < n; i++) pri.set(actionKey(s, p.legal[i]), priors[i]);
    node.priors = pri;
    node.children = new Map();
    this.backup(p.path, chooser === 0 ? v : -v, movesLeft > 0 ? movesLeft : 0);
  }

  /** value0: value for player 0; movesLeft: actions left at the leaf (the last node of `path`). */
  private backup(path: Node[], value0: number, movesLeft: number): void {
    const leaf = path.length - 1; // each tree edge is one action
    for (let i = 1; i < path.length; i++) {
      const nd = path[i];
      nd.N++;
      nd.W += nd.mover === 0 ? value0 : -value0;
      nd.M += movesLeft + leaf - i;
    }
    this.root.N++;
    this.root.W += this.rootState.current === 0 ? value0 : -value0;
    this.root.M += movesLeft + leaf;
  }

  private addNoise(priors: Float32Array, n: number): void {
    const a = this.opts.dirichletAlpha;
    const eps = this.opts.dirichletWeight;
    let sumP = 0;
    for (let i = 0; i < n; i++) sumP += priors[i];
    const g = new Float64Array(n);
    let sumG = 0;
    for (let i = 0; i < n; i++) sumG += g[i] = sampleGamma(a, this.rng);
    for (let i = 0; i < n; i++) priors[i] = (1 - eps) * (priors[i] / (sumP || 1)) + eps * (g[i] / sumG);
  }

  get cPuct(): number {
    return this.opts.cPuct;
  }

  /**
   * Keep the search for the next decision: re-root at the node reached by `keys` (actionKey of each
   * action played since this tree's root, computed before applying it) with `newRoot` the position
   * now. The tree must belong to the player to move at `newRoot` (single observer), and chance
   * events in between are averaged over in the kept statistics. With `dirichletAlpha`, fresh root
   * noise is mixed into the kept priors (self-play). Returns the visits kept, or -1 when that
   * position was never expanded (start a new search instead).
   */
  descend(keys: number[], newRoot: GameState, o: { dirichletAlpha?: number; rootActions?: number[] } = {}): number {
    let node: Node | undefined = this.root;
    for (const k of keys) {
      node = node.children?.get(k);
      if (!node) return -1;
    }
    if (!node.expanded || node.N === 0) return -1;
    // a root keeps W for the player to move there, other nodes for the player who moved into them
    if (node !== this.root && node.mover !== newRoot.current) node.W = -node.W;
    node.mover = 1 - newRoot.current;
    this.root = node;
    this.rootState = newRoot;
    this.rootAllowed = o.rootActions ? new Set(o.rootActions) : null;
    // children of moves that are not legal here (reached in other determinizations) must not be chosen
    const n = this.legalAt(newRoot, true);
    const legalKeys = new Set<number>();
    for (let i = 0; i < n; i++) legalKeys.add(actionKey(newRoot, this.legal[i]));
    for (const k of [...node.children!.keys()]) if (!legalKeys.has(k)) node.children!.delete(k);
    const pri = node.priors!;
    let sum = 0;
    for (const p of pri.values()) sum += p;
    this.rootPriors = new Map([...pri].map(([k, p]) => [k, p / (sum || 1)]));
    if (o.dirichletAlpha && o.dirichletAlpha > 0) {
      const keysAt = [...pri.keys()];
      const ps = new Float32Array(keysAt.map((k) => this.rootPriors!.get(k)!));
      const saved = this.opts.dirichletAlpha;
      this.opts.dirichletAlpha = o.dirichletAlpha;
      this.addNoise(ps, ps.length);
      this.opts.dirichletAlpha = saved;
      keysAt.forEach((k, i) => pri.set(k, ps[i]));
    }
    return node.N;
  }

  result(): SearchResult {
    const children: ChildStat[] = [];
    for (const [key, ch] of this.root.children ?? [])
      children.push({ action: ch.action, key, N: ch.N, Q: ch.N ? ch.W / ch.N : 0, prior: ch.prior, M: ch.N ? ch.M / ch.N : 0 });
    children.sort((a, b) => b.N - a.N || b.Q - a.Q);
    const pv: number[] = [];
    let node: Node | undefined = this.root;
    while (node && node.children && node.children.size > 0 && pv.length < 12) {
      let best: Node | undefined;
      for (const ch of node.children.values()) if (!best || ch.N > best.N) best = ch;
      if (!best || best.N === 0) break;
      pv.push(best.action);
      node = best;
    }
    return {
      action: children[0]?.action ?? -1,
      value: this.root.N ? this.root.W / this.root.N : 0,
      iterations: this.root.N,
      children,
      pv,
    };
  }
}

/** Merge root statistics of several independent trees (root parallelism). */
export function mergeResults(results: SearchResult[]): SearchResult {
  const byKey = new Map<number, ChildStat>();
  let iters = 0;
  let vsum = 0;
  for (const r of results) {
    iters += r.iterations;
    vsum += r.value * r.iterations;
    for (const c of r.children) {
      const m = byKey.get(c.key);
      if (m) {
        m.Q = (m.Q * m.N + c.Q * c.N) / (m.N + c.N || 1);
        m.M = ((m.M ?? 0) * m.N + (c.M ?? 0) * c.N) / (m.N + c.N || 1);
        m.N += c.N;
      } else byKey.set(c.key, { ...c });
    }
  }
  const children = [...byKey.values()].sort((a, b) => b.N - a.N || b.Q - a.Q);
  const lead = results.find((r) => r.action === children[0]?.action) ?? results[0];
  return { action: children[0]?.action ?? -1, value: iters ? vsum / iters : 0, iterations: iters, children, pv: lead?.pv ?? [] };
}
