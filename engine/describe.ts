// Human-readable (Chinese) descriptions of engine actions, for the local app.
import { LINES, rowOf, colOf } from "./board";
import { CARDS, ROYALS } from "./cards";
import {
  A_BUY,
  A_CELL,
  A_DISCARD,
  A_JOKER,
  A_PASS,
  A_REPLENISH,
  A_RESERVE,
  A_ROYAL,
  A_STEAL,
  A_TAKE_LINE,
  NUM_LINES,
  Phase,
  TOKEN_NAMES,
} from "./constants";
import { slotLevel } from "./rules";
import { GameState } from "./state";

export function cellName(cell: number): string {
  return `(${rowOf(cell)},${colOf(cell)})`;
}

export function cardName(id: number): string {
  const c = CARDS[id];
  const color = c.joker ? "百搭" : c.color === 9 ? "无色" : TOKEN_NAMES[c.color];
  const parts = [`L${c.level}${color}卡`];
  if (c.points) parts.push(`${c.points}分`);
  if (c.crowns) parts.push(`${c.crowns}冠`);
  if (c.bonus > 1) parts.push(`${c.bonus}奖励`);
  const ab = [c.again && "再来一回合", c.token && "拿同色币", c.privilege && "拿特权", c.steal && "偷币"].filter(Boolean);
  if (ab.length) parts.push(ab.join("+"));
  return parts.join(" ");
}

export function royalName(id: number): string {
  const r = ROYALS[id];
  const ab = r.again ? "+再来一回合" : r.privilege ? "+特权" : r.steal ? "+偷币" : "";
  return `王室卡 ${r.points}分${ab}`;
}

/** Describe action `a` as taken in state `s` (before applying it). */
export function describeAction(s: GameState, a: number): string {
  const p = s.players[s.current];
  if (a >= A_TAKE_LINE && a < A_TAKE_LINE + NUM_LINES) {
    const cells = LINES[a - A_TAKE_LINE];
    return `拿币 ${cells.map((c) => `${TOKEN_NAMES[s.board[c]]}${cellName(c)}`).join(" ")}`;
  }
  if (a >= A_CELL && a < A_CELL + 25) {
    const c = a - A_CELL;
    const tok = `${TOKEN_NAMES[s.board[c]]}${cellName(c)}`;
    if (s.phase === Phase.ReserveGold) return `拿金币 ${cellName(c)}`;
    if (s.phase === Phase.TakeToken) return `能力：拿 ${tok}`;
    return `用特权拿 ${tok}`;
  }
  if (a === A_REPLENISH) return "补充棋盘（对手得 1 特权）";
  if (a >= A_RESERVE && a < A_RESERVE + 15) {
    const t = a - A_RESERVE;
    if (t >= 12) return `预留 L${t - 11} 牌堆顶（盲抽）`;
    return `预留 ${cardName(s.pyramid[t])}（第${slotLevel(t)}层）`;
  }
  if (a >= A_BUY && a < A_BUY + 15) {
    const src = a - A_BUY;
    const id = src < 12 ? s.pyramid[src] : p.reserved[src - 12];
    return `购买 ${cardName(id)}${src >= 12 ? "（预留区）" : ""}`;
  }
  if (a >= A_JOKER && a < A_JOKER + 5) return `百搭卡放到 ${TOKEN_NAMES[a - A_JOKER + 1]} 列`;
  if (a >= A_STEAL && a < A_STEAL + 6) return `偷对手 1 个${TOKEN_NAMES[a - A_STEAL]}`;
  if (a >= A_ROYAL && a < A_ROYAL + 4) return `选 ${royalName(a - A_ROYAL)}`;
  if (a >= A_DISCARD && a < A_DISCARD + 7) return `弃 1 个${TOKEN_NAMES[a - A_DISCARD]}`;
  if (a === A_PASS) return "无合法动作，跳过";
  return `动作 ${a}`;
}
