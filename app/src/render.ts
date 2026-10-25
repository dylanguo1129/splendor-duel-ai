// DOM rendering of a GameState. Pure view: callbacks are passed in by the controller.
import { CARDS, ROYALS } from "../../engine/cards";
import { EMPTY, GOLD, LEVEL_SLOT_START, Phase, TOKEN_NAMES } from "../../engine/constants";
import { GameState, tokenCount } from "../../engine/state";

export interface Highlights {
  cells: Set<number>;
  slots: Set<number>; // pyramid slots 0..11
  decks: Set<number>; // levels 1..3
  reserved: Set<number>; // reserved index of the player to move
  royals: Set<number>;
}

export const noHighlights = (): Highlights => ({ cells: new Set(), slots: new Set(), decks: new Set(), reserved: new Set(), royals: new Set() });

const h = (tag: string, cls = "", text = ""): HTMLElement => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text) e.textContent = text;
  return e;
};

export function tokenChip(type: number, count?: number): HTMLElement {
  const e = h("span", `tok t${type}`);
  e.title = TOKEN_NAMES[type];
  if (count !== undefined) e.textContent = String(count);
  return e;
}

export function cardEl(id: number, opts: { faceDown?: boolean; level?: number; highlight?: boolean; affordable?: boolean } = {}): HTMLElement {
  if (opts.faceDown) {
    const e = h("div", `card back l${opts.level}`);
    e.append(h("div", "back-label", `L${opts.level}`));
    if (opts.highlight) e.classList.add("hl");
    return e;
  }
  const c = CARDS[id];
  const cls = c.joker ? "cj" : c.color === 9 ? "cx" : `c${c.color}`;
  const e = h("div", `card ${cls}`);
  if (opts.highlight) e.classList.add("hl");
  if (opts.affordable) e.classList.add("afford");
  const top = h("div", "card-top");
  top.append(h("span", "pts", c.points ? String(c.points) : ""));
  if (c.crowns) top.append(h("span", "crowns", "♛".repeat(c.crowns)));
  const bonus = h("span", "bonus");
  if (c.bonus) bonus.append(...Array.from({ length: c.bonus }, () => h("span", `gem ${c.joker ? "gj" : `g${c.color}`}`)));
  top.append(bonus);
  e.append(top);
  const ab = [c.again && "再来", c.token && "拿币", c.privilege && "特权", c.steal && "偷币", c.joker && "百搭"].filter(Boolean).join(" ");
  e.append(h("div", "ability", ab));
  const cost = h("div", "cost");
  for (let t = 0; t <= 5; t++) if (c.cost[t]) cost.append(tokenChip(t, c.cost[t]));
  e.append(cost);
  return e;
}

export function renderBoard(el: HTMLElement, s: GameState, sel: Set<number>, hl: Highlights, onCell: (c: number) => void): void {
  el.replaceChildren();
  for (let cell = 0; cell < 25; cell++) {
    const sq = h("div", "cell");
    const t = s.board[cell];
    if (t !== EMPTY) {
      const tok = tokenChip(t);
      tok.classList.add("big");
      sq.append(tok);
    }
    if (sel.has(cell)) sq.classList.add("sel");
    if (hl.cells.has(cell)) sq.classList.add("hl");
    sq.onclick = () => onCell(cell);
    el.append(sq);
  }
}

export function renderPyramid(
  el: HTMLElement,
  s: GameState,
  viewer: number,
  hl: Highlights,
  on: { buy: (slot: number) => void; reserve: (target: number) => void },
  affordable: (id: number) => boolean,
): void {
  el.replaceChildren();
  for (const level of [3, 2, 1]) {
    const row = h("div", "pyr-row");
    const deck = h("div", "deck");
    deck.append(h("div", "deck-count", `L${level} 牌堆 ${s.decks[level].length}`));
    if (hl.decks.has(level)) deck.classList.add("hl");
    deck.onclick = () => on.reserve(11 + level);
    row.append(deck);
    const n = level === 1 ? 5 : level === 2 ? 4 : 3;
    for (let k = 0; k < n; k++) {
      const slot = LEVEL_SLOT_START[level] + k;
      const id = s.pyramid[slot];
      const wrap = h("div", "slot");
      if (id === EMPTY) wrap.append(h("div", "card empty"));
      else {
        const ce = cardEl(id, { highlight: hl.slots.has(slot), affordable: affordable(id) });
        ce.onclick = () => on.buy(slot);
        wrap.append(ce);
        const rb = h("button", "mini", "预留");
        rb.onclick = (ev) => {
          ev.stopPropagation();
          on.reserve(slot);
        };
        wrap.append(rb);
      }
      row.append(wrap);
    }
    el.append(row);
  }
  void viewer;
}

export function renderRoyals(el: HTMLElement, s: GameState, hl: Highlights): void {
  el.replaceChildren(h("div", "label", "王室卡"));
  for (const r of s.royalsAvail) {
    const R = ROYALS[r];
    const e = h("div", "royal", `${R.points}分 ${R.again ? "再来" : R.privilege ? "特权" : R.steal ? "偷币" : ""}`);
    if (hl.royals.has(r)) e.classList.add("hl");
    el.append(e);
  }
}

export function renderPlayer(
  el: HTMLElement,
  s: GameState,
  pi: number,
  opts: { name: string; viewer: number; reveal: boolean; hl: Highlights; onReserved: (r: number) => void; affordable: (id: number) => boolean },
): void {
  const p = s.players[pi];
  el.replaceChildren();
  el.classList.toggle("active", s.current === pi && s.phase !== Phase.GameOver);
  const head = h("div", "p-head");
  head.append(h("span", "p-name", opts.name));
  let maxCol = 0;
  for (let c = 1; c <= 5; c++) maxCol = Math.max(maxCol, p.colorPoints[c]);
  head.append(
    h("span", "stat", `分 ${p.points}/20`),
    h("span", "stat", `冠 ${p.crowns}/10`),
    h("span", "stat", `单色 ${maxCol}/10`),
    h("span", "stat", `特权 ${p.privileges}`),
    h("span", "stat", `币 ${tokenCount(p)}/10`),
  );
  el.append(head);
  const cols = h("div", "p-cols");
  for (const color of [1, 2, 3, 4, 5]) {
    const col = h("div", "p-col");
    const b = h("div", `bonus-box g${color}`, String(p.bonus[color]));
    b.title = `${TOKEN_NAMES[color]} 奖励（该色卡分数 ${p.colorPoints[color]}）`;
    col.append(b, h("div", "colpts", p.colorPoints[color] ? `${p.colorPoints[color]}分` : ""), tokenChip(color, p.tokens[color]));
    cols.append(col);
  }
  const extra = h("div", "p-col");
  extra.append(tokenChip(0, p.tokens[0]), tokenChip(GOLD, p.tokens[GOLD]));
  cols.append(extra);
  el.append(cols);
  const res = h("div", "p-reserved");
  p.reserved.forEach((id, r) => {
    const hidden = p.reservedHidden[r] && pi !== opts.viewer && !opts.reveal;
    const ce = cardEl(id, { faceDown: hidden, level: CARDS[id].level, highlight: s.current === pi && opts.hl.reserved.has(r), affordable: !hidden && opts.affordable(id) });
    ce.onclick = () => opts.onReserved(r);
    res.append(ce);
  });
  if (p.royals.length) res.append(...p.royals.map((r) => h("div", "royal owned", `${ROYALS[r].points}分`)));
  el.append(res);
}
