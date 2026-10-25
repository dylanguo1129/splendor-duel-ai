import { SearchPool, SearchTiming } from "../../ai/search-pool";
import type { SearchResult } from "../../ai/mcts";
import type { NetJson } from "../../ai/net";
import { Outlook, summarizeOutlook } from "../../ai/outlook";
import { outlookText, timingText } from "../../ai/describe-search";
import { lineIndexOf } from "../../engine/board";
import { A_BUY, A_CELL, A_RESERVE, A_ROYAL, A_TAKE_LINE, DRAW, NUM_LINES, Phase } from "../../engine/constants";
import { LINES } from "../../engine/board";
import { describeAction } from "../../engine/describe";
import { determinize } from "../../engine/determinize";
import { Rng } from "../../engine/rng";
import { applyAction, canAfford, listActions } from "../../engine/rules";
import { stateFromJson, stateToJson } from "../../engine/serialize";
import { cloneState, GameState, newGame } from "../../engine/state";
import { Highlights, noHighlights, renderBoard, renderPlayer, renderPyramid, renderRoyals } from "./render";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const WIN_REASON = ["", "20 分", "10 皇冠", "单色 10 分", "囤积判负", "平局"];

let net: NetJson | null = null;
try {
  const r = await fetch("./models/best.json");
  if (r.ok) net = await r.json();
} catch {
  net = null;
}
const threads = Math.max(1, Math.min(8, (navigator.hardwareConcurrency || 4) - 1));
const pool = new SearchPool(() => new Worker(new URL("../../ai/search-worker.ts", import.meta.url), { type: "module" }), threads, net);
const engineName = net ? `神经网络 ${(net.meta as any)?.name ?? ""}`.trim() : "启发式 MCTS";

let state: GameState = newGame(Date.now() >>> 0);
let history: GameState[] = [];
let rng = new Rng(Date.now() >>> 0);
let selected = new Set<number>();
let hint: { forPly: number; result: SearchResult; timing: SearchTiming | null; outlook: Outlook | null } | null = null;
let thinking = false;
let epoch = 0; // invalidates in-flight searches on new game / undo

const mode = () => $<HTMLSelectElement>("mode").value as "hva" | "ava" | "analysis";
const humanSeat = () => Number($<HTMLSelectElement>("human-seat").value);
/** Search with the selected thinking time ("auto:<ms>" = adaptive, see ai/time-control.ts). */
async function think(s: GameState): Promise<{ result: SearchResult; timing: SearchTiming | null }> {
  const v = $<HTMLSelectElement>("think").value;
  const ms = Number(v.replace("auto:", ""));
  if (!v.startsWith("auto:")) return { result: await pool.search(s, { timeMs: ms }), timing: null };
  const r = await pool.searchAuto(s, { base: ms });
  return { result: r, timing: r.timing };
}

/** Fill in the turns-to-win estimate of the current hint once the playouts finish. */
async function addOutlook(h: NonNullable<typeof hint>, s: GameState): Promise<void> {
  if (Number.isNaN(h.result.value)) return;
  const mover = s.current; // `state` is mutated in place once a move is played
  const runs = await pool.playouts(s, 64, 600);
  if (hint !== h) return;
  h.outlook = summarizeOutlook(runs, mover, (h.result.value + 1) / 2);
  render();
}
const isAiTurn = () => state.phase !== Phase.GameOver && (mode() === "ava" || (mode() === "hva" && state.current !== humanSeat()));
const viewer = () => (mode() === "hva" ? humanSeat() : state.current);

function highlightsFor(s: GameState, a: number | undefined): Highlights {
  const hl = noHighlights();
  if (a === undefined) return hl;
  if (a >= A_TAKE_LINE && a < A_TAKE_LINE + NUM_LINES) LINES[a - A_TAKE_LINE].forEach((c) => hl.cells.add(c));
  else if (a >= A_CELL && a < A_CELL + 25) hl.cells.add(a - A_CELL);
  else if (a >= A_RESERVE && a < A_RESERVE + 15) {
    const t = a - A_RESERVE;
    if (t < 12) hl.slots.add(t);
    else hl.decks.add(t - 11);
  } else if (a >= A_BUY && a < A_BUY + 15) {
    const src = a - A_BUY;
    if (src < 12) hl.slots.add(src);
    else hl.reserved.add(src - 12);
  } else if (a >= A_ROYAL && a < A_ROYAL + 4) hl.royals.add(a - A_ROYAL);
  void s;
  return hl;
}

function play(a: number): void {
  if (!listActions(state).includes(a)) return;
  history.push(cloneState(state));
  applyAction(state, a, rng);
  selected.clear();
  hint = null;
  step();
}

function tryAction(a: number): void {
  if (isAiTurn() || thinking) return;
  play(a);
}

function onCell(cell: number): void {
  if (isAiTurn()) return;
  const legal = listActions(state);
  if (state.phase !== Phase.Main) {
    if (legal.includes(A_CELL + cell)) play(A_CELL + cell);
    return;
  }
  if (selected.has(cell)) selected.delete(cell);
  else selected.add(cell);
  render();
}

function selectionActions(): { label: string; action: number }[] {
  const out: { label: string; action: number }[] = [];
  if (state.phase !== Phase.Main || selected.size === 0) return out;
  const legal = listActions(state);
  const cells = [...selected];
  const li = lineIndexOf(cells);
  if (li >= 0 && legal.includes(A_TAKE_LINE + li)) out.push({ label: `拿取所选 ${cells.length} 个`, action: A_TAKE_LINE + li });
  if (cells.length === 1 && legal.includes(A_CELL + cells[0])) out.push({ label: "用 1 个特权拿这个", action: A_CELL + cells[0] });
  return out;
}

function renderAiPanel(): void {
  const el = $("ai-panel");
  el.replaceChildren();
  const head = document.createElement("div");
  head.className = "ai-head";
  head.innerHTML = `<b>AI</b> <span class="muted">${engineName} · ${threads} 线程</span>`;
  const btn = document.createElement("button");
  btn.textContent = thinking ? "思考中…" : "提示";
  btn.disabled = thinking || state.phase === Phase.GameOver;
  btn.onclick = () => requestHint();
  head.append(btn);
  el.append(head);
  if (!hint || hint.forPly !== state.ply) return;
  const r = hint.result;
  if (!Number.isNaN(r.value)) {
    const pct = Math.round(((r.value + 1) / 2) * 100);
    const bar = document.createElement("div");
    bar.className = "winbar";
    bar.innerHTML = `<div class="fill" style="width:${pct}%"></div><span>${state.current === viewer() ? "我方" : "对方"}胜率 ${pct}%</span>`;
    el.append(bar);
  }
  if (hint.outlook) {
    const mine = state.current === viewer();
    el.append(Object.assign(document.createElement("div"), { className: "outlook", textContent: outlookText(hint.outlook, state.current, mine ? "我方" : "对方", mine ? "对方" : "我方") }));
  }
  const rec = document.createElement("div");
  rec.className = "rec";
  // describe the principal variation for the rest of this turn on a determinization (no peeking)
  const d = determinize(state, state.current, new Rng(1));
  const steps: string[] = [];
  const me = d.current;
  for (const a of r.pv) {
    if (d.phase === Phase.GameOver || d.current !== me || !listActions(d).includes(a)) break;
    steps.push(describeAction(d, a));
    const turn = d.turn;
    applyAction(d, a, new Rng(2));
    if (d.turn !== turn) break;
  }
  rec.innerHTML = `<div class="label">推荐</div>${steps.map((x, i) => `<div class="step">${i + 1}. ${x}</div>`).join("")}`;
  el.append(rec);
  const total = r.children.reduce((a, c) => a + c.N, 0) || 1;
  const alt = document.createElement("table");
  alt.className = "alts";
  alt.innerHTML = "<tr><th>候选</th><th>访问</th><th>胜率</th></tr>";
  for (const c of r.children.slice(0, 6)) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${describeAction(state, c.action)}</td><td>${Math.round((100 * c.N) / total)}%</td><td>${Number.isNaN(c.Q) ? "-" : Math.round(((c.Q + 1) / 2) * 100) + "%"}</td>`;
    tr.onclick = () => tryAction(c.action);
    alt.append(tr);
  }
  el.append(alt);
  el.append(
    Object.assign(document.createElement("div"), {
      className: "muted small",
      textContent: `搜索 ${r.iterations} 次模拟${hint.timing ? ` · ${timingText(hint.timing)}` : ""}`,
    }),
  );
}

function render(): void {
  const v = viewer();
  const reveal = $<HTMLInputElement>("reveal").checked;
  const hl = hint && hint.forPly === state.ply ? highlightsFor(state, hint.result.action) : noHighlights();
  const me = state.players[state.current];
  const affordable = (id: number) => canAfford(me, id);
  renderBoard($("board"), state, selected, hl, onCell);
  renderPyramid($("pyramid"), state, v, hl, { buy: (slot) => tryAction(A_BUY + slot), reserve: (t) => tryAction(A_RESERVE + t) }, affordable);
  renderRoyals($("royals"), state, hl);
  const names = mode() === "hva" ? (v === 0 ? ["我方", "AI"] : ["AI", "我方"]) : ["玩家 1", "玩家 2"];
  const opp = 1 - v;
  for (const [elId, pi] of [["opponent", opp], ["me", v]] as const)
    renderPlayer($(elId), state, pi, {
      name: names[pi] + (state.firstPlayer === pi ? "（先手）" : ""),
      viewer: v,
      reveal,
      hl,
      onReserved: (r) => tryAction(A_BUY + 12 + r),
      affordable: (id) => canAfford(state.players[pi], id),
    });
  const info = $("board-info");
  info.textContent = `袋中 ${state.bag.reduce((a, b) => a + b, 0)} 枚 · 桌面特权 ${state.tablePrivileges} · 第 ${Math.ceil(state.turn / 2) + 1} 轮`;
  const ba = $("board-actions");
  ba.replaceChildren();
  for (const sa of selectionActions()) {
    const b = document.createElement("button");
    b.className = "primary";
    b.textContent = sa.label;
    b.onclick = () => tryAction(sa.action);
    ba.append(b);
  }
  if (selected.size) {
    const b = document.createElement("button");
    b.textContent = "清除选择";
    b.onclick = () => {
      selected.clear();
      render();
    };
    ba.append(b);
  }
  const status = $("status");
  if (state.phase === Phase.GameOver) {
    status.textContent =
      state.winner === DRAW ? "平局" : `${names[state.winner]} 获胜（${WIN_REASON[state.winReason]}）`;
    status.className = "status over";
  } else {
    const phaseName = ["主动作", "选金币", "百搭卡放哪列", "能力：拿币", "偷币", "选王室卡", "弃币", ""][state.phase];
    status.textContent = `轮到 ${names[state.current]} · ${phaseName}${thinking ? " · AI 思考中…" : ""}`;
    status.className = "status";
  }
  const acts = $("actions");
  acts.replaceChildren();
  if (!isAiTurn() && state.phase !== Phase.GameOver) {
    for (const a of listActions(state)) {
      const b = document.createElement("button");
      b.className = "act" + (hint && hint.forPly === state.ply && hint.result.action === a ? " best" : "");
      b.textContent = describeAction(state, a);
      b.onclick = () => tryAction(a);
      acts.append(b);
    }
  }
  renderAiPanel();
}

async function requestHint(): Promise<void> {
  if (thinking || state.phase === Phase.GameOver) return;
  thinking = true;
  const my = epoch;
  const ply = state.ply;
  render();
  const { result, timing } = await think(state);
  thinking = false;
  if (my !== epoch || ply !== state.ply) return render();
  hint = { forPly: ply, result, timing, outlook: null };
  render();
  void addOutlook(hint, state);
}

async function step(): Promise<void> {
  render();
  if (state.phase === Phase.GameOver) return;
  if (isAiTurn()) {
    if (thinking) return;
    thinking = true;
    const my = epoch;
    render();
    const { result, timing } = await think(state);
    thinking = false;
    if (my !== epoch) return;
    hint = { forPly: state.ply, result, timing, outlook: null };
    render();
    void addOutlook(hint, state);
    await new Promise((r) => setTimeout(r, mode() === "ava" ? 250 : 400));
    if (my !== epoch) return;
    play(result.action);
  } else if ($<HTMLInputElement>("auto-hint").checked && mode() !== "analysis") requestHint();
}

function newGameClick(): void {
  epoch++;
  thinking = false;
  state = newGame(Date.now() >>> 0, 0);
  rng = new Rng(Date.now() >>> 0);
  history = [];
  selected.clear();
  hint = null;
  step();
}

$("new-game").onclick = newGameClick;
$("undo").onclick = () => {
  if (!history.length) return;
  epoch++;
  thinking = false;
  // undo back to the previous human decision
  do state = history.pop()!;
  while (history.length && isAiTurn());
  hint = null;
  selected.clear();
  step();
};
for (const id of ["mode", "human-seat", "reveal", "auto-hint"]) $(id).onchange = () => step();
$("load-json").onclick = () => {
  try {
    state = stateFromJson(JSON.parse($<HTMLTextAreaElement>("state-json").value));
    epoch++;
    thinking = false;
    history = [];
    hint = null;
    $<HTMLSelectElement>("mode").value = "analysis";
    step();
  } catch (e) {
    $("status").textContent = `局面 JSON 无法解析：${(e as Error).message}`;
  }
};
$("export-json").onclick = () => {
  $<HTMLTextAreaElement>("state-json").value = JSON.stringify(stateToJson(state));
};

newGameClick();
