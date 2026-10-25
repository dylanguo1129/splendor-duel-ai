# Unofficial AI for Splendor Duel

An unofficial, fan-made AlphaZero-style AI for the two-player board game Splendor Duel (base game): an exact rules
engine, information-set Monte Carlo tree search, a small policy/value network trained by self-play,
and a local web app where you can play against it, watch AI vs AI, or analyse a position.

Everything runs on one PC: training on a consumer GPU, and play in the browser on the CPU
(WebAssembly SIMD inference, a few thousand searches per second per thread).

> Unofficial fan project. Not affiliated with or endorsed by Space Cowboys, Asmodee or
> Board Game Arena. Splendor Duel is a trademark of its owners. See [Scope](#scope).

## What is in here

| Folder | Contents |
|---|---|
| `engine/` | Rules engine in dependency-free TypeScript: state, legal moves, effects, win conditions, observation encoding (864 features from the point of view of the player to move, no hidden information) |
| `ai/` | Single-observer ISMCTS with PUCT, determinization of hidden cards, root-parallel search in Web Workers, network inference (plain JS and WebAssembly SIMD), adaptive thinking time, outlook / turns-to-win estimate |
| `train/` | Generation loop (self-play → train → gate → promote), batched GPU inference server, PyTorch training (`train.py`), match and benchmark tools |
| `app/` | Local web app (Vite): human vs AI, AI vs AI, position analysis |
| `models/best.json` | The final network (generation 91) |
| `tests/` | Vitest suite: rules, conservation checks over random games, search, inference |

## Strength

Round robin, every version against fixed anchors, 50 games per pair at 160 simulations per move with
the same (old) search settings, so only the network differs. Bradley–Terry Elo, the hand-written
heuristic MCTS fixed at 0, 7,072 games in total:

| Version | Elo | What changed |
|---|---|---|
| gen0 | −92 | first network, trained on heuristic self-play |
| gen17 | +302 | 512×512 network, GPU-batched self-play |
| gen51_2 | +455 | win-path features, league self-play, 600-simulation self-play |
| gen74 | +567 | two-stage gate, relative first-play urgency, honest held-out validation |
| gen88 | +677 | entity policy head (per-card scorer) |

The final model (gen91) was promoted twice after gen88. Recent versions win 95–98% against the
heuristic MCTS at equal simulations, and the search settings it ships with (relative FPU) add roughly
+70 Elo at 200 simulations on top of the network-only numbers above.

## Quick start

```bash
npm install
npm run app        # http://localhost:5178
npm test
```

The app uses `models/best.json` when present and falls back to the heuristic MCTS otherwise.

## How it works

- **Search.** Single-observer ISMCTS: every simulation samples the hidden information (deck order,
  the opponent's blind reservations) consistently with what the player to move has seen, so the AI
  never peeks. PUCT selection, first-play urgency relative to the parent's value (0.3 below it),
  and an optional moves-left term that prefers quicker wins when ahead.
- **Network.** MLP 864 → 512 → 512 with heads for policy (224 actions), value, moves left, and a
  training-only auxiliary head (final point / crown / colour differences and how the game ended).
  The **entity policy terms** (`train/entity.py`, mirrored in `ai/net.ts`) score every
  card-buying or card-reserving move with a query shared across pyramid slots, dotted with that
  card's features and each player's per-colour cost shortfall, and score token-taking moves per
  token type. Against the same network without them (same parent, data and recipe) they win
  62% at 200 simulations and 64% at 64.
- **Training loop** (`train/loop.ts`). Each generation: 3,000 self-play games at 600 simulations
  on 35% of moves (playout cap randomisation), half of them against one-track "rush" styles and
  older versions (league); fine-tune the best model on the last 8 generations with whole games held
  out for validation; gate the candidate against the best at equal compute (400 games, ≥53%, or
  50–53% plus 800 more games and ≥51.5% combined).
- **Things that were measured and not kept:** a deeper 3×512 network (weaker at equal compute),
  KataGo-style policy-target pruning (45% vs raw targets), search-tree reuse between moves
  (+ at 200 simulations, − at 600), FPU scaled by visited prior mass, weight averaging.

## Training

Needs Python 3.10+ with CUDA PyTorch and Node 22.

```bash
npm run train -- --gpu 1 --gens 40      # foreground
```

On Windows, `scripts/start-training.ps1` / `scripts/stop-training.ps1` run the loop as a
background process with the settings used for the final model; it resumes from
`train/runs/state.json`. Memory guards abort a phase cleanly instead of exhausting RAM.
Self-play data goes to `train/data/` (only recent generations are kept).

Useful tools: `node dist/train/match.mjs <specA> <specB> --gpu 1` for head-to-head matches
(spec options: `sims=`, `mlh`, `fpur=`, `c=`, `style=` …), `python train/train.py --eval-only`
for held-out losses, `npx tsx scripts/check-net-parity.ts` to compare the browser inference with
PyTorch.

## Scope

This is a non-commercial research and hobby project. It contains **no tool that connects to,
plays on or advises in any online game platform**; use the AI offline, for practice and study.
Online platforms such as Board Game Arena forbid outside assistance in games against other people;
please respect that.

`data/cards.json` holds the card statistics (costs, points, crowns, abilities) that the engine needs
to implement the rules; it contains no card text or artwork. Splendor Duel, its name, rules text and
artwork belong to their owners (designed by Marc André and Bruno Cathala, published by Space
Cowboys). If you represent the rights holders and want anything changed or removed, please open an
issue and it will be done promptly.

## License

MIT, see [LICENSE](LICENSE).

---

## 中文简介

双人桌游 **Splendor Duel（璀璨宝石·对决）** 基础版的 AlphaZero 式 AI：精确的规则引擎、信息集蒙特卡洛树搜索、自博弈训练的小型策略/价值网络，以及可以和 AI 对战、看 AI 互搏、分析局面的本地网页版。训练在一张消费级显卡上完成，下棋在浏览器里用 CPU（WebAssembly SIMD）运行。

本仓库是非商业的研究和兴趣项目，**不包含任何连接在线对战平台、在线上对局中下棋或给建议的工具**，请离线用于练习和学习。非官方粉丝项目，与 Space Cowboys、Asmodee、Board Game Arena 无关；版权方如需修改或删除任何内容，请提 issue。
