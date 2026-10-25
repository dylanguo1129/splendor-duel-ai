import { CARDS } from "./cards";
import { Rng } from "./rng";
import { cloneState, GameState } from "./state";

/**
 * Hidden information from `observer`'s point of view: deck order and the opponent's
 * blind reservations (only their level is known). Returns a clone where those are
 * resampled uniformly from the cards the observer cannot see.
 */
export function determinize(s: GameState, observer: number, rng: Rng, resampleOwnHidden = false): GameState {
  const d = cloneState(s);
  const who = resampleOwnHidden ? [0, 1] : [1 - observer];
  for (let level = 1; level <= 3; level++) {
    const pool = d.decks[level].slice();
    const hiddenSlots: [number, number][] = [];
    for (const pi of who) {
      const p = d.players[pi];
      p.reserved.forEach((id, i) => {
        if (p.reservedHidden[i] && CARDS[id].level === level) {
          hiddenSlots.push([pi, i]);
          pool.push(id);
        }
      });
    }
    if (pool.length === 0) continue;
    rng.shuffle(pool);
    for (const [pi, i] of hiddenSlots) d.players[pi].reserved[i] = pool.pop()!;
    d.decks[level] = pool;
  }
  return d;
}

/** Cards `observer` cannot identify (deck contents + opponent's blind reserves), by level. */
export function unknownCards(s: GameState, observer: number): number[][] {
  const out: number[][] = [[], [], [], []];
  for (let level = 1; level <= 3; level++) out[level].push(...s.decks[level]);
  const opp = s.players[1 - observer];
  opp.reserved.forEach((id, i) => {
    if (opp.reservedHidden[i]) out[CARDS[id].level].push(id);
  });
  return out;
}
