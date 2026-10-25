// JSON (de)serialization of GameState (typed arrays become plain arrays).
import { emptyState, GameState, PlayerState } from "./state";

type Json = Record<string, unknown>;

function plainPlayer(p: PlayerState): Json {
  return { ...p, tokens: [...p.tokens], bonus: [...p.bonus], colorPoints: [...p.colorPoints] };
}

export function stateToJson(s: GameState): Json {
  return {
    ...s,
    board: [...s.board],
    bag: [...s.bag],
    pyramid: [...s.pyramid],
    owner: [...s.owner],
    hoard: [...s.hoard],
    players: s.players.map(plainPlayer),
  };
}

export function stateFromJson(j: any): GameState {
  const s = emptyState();
  const typed = ["board", "bag", "pyramid", "owner", "hoard"] as const;
  for (const [k, v] of Object.entries(j)) {
    if ((typed as readonly string[]).includes(k)) (s as any)[k].set(v as number[]);
    else if (k !== "players") (s as any)[k] = structuredClone(v);
  }
  j.players.forEach((pj: any, i: number) => {
    const p = s.players[i];
    for (const [k, v] of Object.entries(pj)) {
      if (k === "tokens" || k === "bonus" || k === "colorPoints") (p as any)[k].set(v as number[]);
      else (p as any)[k] = structuredClone(v);
    }
  });
  return s;
}
