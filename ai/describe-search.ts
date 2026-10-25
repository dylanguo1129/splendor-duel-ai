// Chinese UI text for auto-mode timing and the turns-to-win estimate (local app).
import type { Outlook } from "./outlook";
import type { SearchTiming } from "./search-pool";

const REASON: Record<SearchTiming["reason"], string> = {
  forced: "只有一种走法",
  decided: "最优解已确定，提前结束",
  target: "按计划",
  extended: "候选很接近，多想了一会",
  max: "达到上限",
};

/** Ways of winning by engine WIN_REASON_* code. */
const PATH = ["", "拿到 20 分", "集齐 10 皇冠", "单色 10 分", "对方囤积判负"];

export function timingText(t: SearchTiming): string {
  return `用时 ${(t.spentMs / 1000).toFixed(1)} 秒（${REASON[t.reason]}）`;
}

/** `me`/`opp` name the player to move and the other one ("你"/"对手", or "行动方"/"另一方"). */
export function outlookText(o: Outlook, mover: number, me: string, opp: string): string {
  const name = (pl: number) => (pl === mover ? me : opp);
  const via = (pl: number) => (o.paths[pl]?.reason ? `，多半靠${PATH[o.paths[pl].reason]}` : "");
  const range = o.lo === o.hi ? "" : `大致 ${o.lo}–${o.hi} 回合`;
  if (o.close) {
    const both = [mover, 1 - mover].filter((pl) => o.paths[pl]?.reason).map((pl) => `${name(pl)}多半靠${PATH[o.paths[pl].reason]}`);
    return `势均力敌，预计${me}还要约 ${o.turns} 个回合分出胜负（含本回合${range ? `，${range}` : ""}）${both.length ? `；${both.join("，")}` : ""}`;
  }
  // the other side's best chance: its most frequent way of winning the playouts
  const other = 1 - o.side;
  const alt = o.paths[other];
  const chance = alt?.reason && alt.share >= 0.15 ? `；${name(other)}${other === mover ? "翻盘路线" : "的威胁"}：${PATH[alt.reason]}（约 ${Math.round(alt.share * 100)}%）` : "";
  if (o.side === mover) {
    if (o.turns <= 1) return `${me}本回合就能获胜${via(mover)}`;
    return `${me}预计还要约 ${o.turns} 个回合获胜${via(mover)}（含本回合${range ? `，${range}` : ""}）${chance}`;
  }
  return `${opp}预计再走约 ${o.turns} 个回合获胜${via(o.side)}${range ? `（${range}）` : ""}${chance}`;
}
