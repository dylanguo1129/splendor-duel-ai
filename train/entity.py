"""Entity policy terms, shared by train/train.py (PVNet) and train/stacked.py (GPU batching).

Logits of moves on a card or on board cells get a score shared across slots: a state query q(h)
dotted with that card's features (+ its per-colour cost shortfall for each player), or a
per-token-type score summed over the cells taken. ai/net.ts computes the same terms in the browser.
The ENT_OUT query outputs are [buy card 40 | reserve card 40 | line cell by token type 7 | single
cell by token type 7].
"""
import torch

NUM_ACTIONS = 224  # engine/constants.ts
CARD_DIM = 28 + 12  # card block (engine/features.ts encodeCard) + shortfall per colour, me and opponent
ENT_OUT = 2 * CARD_DIM + 14
ME_BLOCK, OPP_BLOCK, PYRAMID, MY_RESERVED = 183, 213, 243, 579  # feature offsets (engine/features.ts)


def line_matrix():
    """[145, 25] cell membership of each take-line action, in engine/board.ts LINES order: 25 singles,
    then pairs, then triples, each by row, column and direction (0,1), (1,0), (1,1), (1,-1)."""
    lines = [[c] for c in range(25)]
    for ln in (2, 3):
        for r in range(1, 6):
            for c in range(1, 6):
                for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                    cells = [(r + dr * k - 1) * 5 + c + dc * k - 1 for k in range(ln) if 1 <= r + dr * k <= 5 and 1 <= c + dc * k <= 5]
                    if len(cells) == ln:
                        lines.append(cells)
    assert len(lines) == 145
    M = torch.zeros(145, 25)
    for i, cells in enumerate(lines):
        M[i, cells] = 1
    return M


def entity_logits(x, q, M):
    """Entity terms of the policy logits. x: [..., F] features, q: [..., ENT_OUT], M: line_matrix().
    Returns [..., NUM_ACTIONS] (zero for moves without a card or cell)."""
    lead = x.shape[:-1]
    xf = x.float()
    cards = torch.cat([xf[..., PYRAMID : PYRAMID + 12 * 28].reshape(*lead, 12, 28), xf[..., MY_RESERVED : MY_RESERVED + 3 * 28].reshape(*lead, 3, 28)], -2)
    present = (cards[..., 0:1] > 0).float()
    cost = torch.round(cards[..., 11:17] * 8)  # pearl, blue, white, green, black, red

    def shortfall(base):
        tokens = torch.round(xf[..., base : base + 6] * 4)  # same order as the cost
        bonus = torch.cat([torch.zeros_like(tokens[..., :1]), torch.round(xf[..., base + 8 : base + 13] * 5)], -1)
        return torch.clamp(cost - (tokens + bonus).unsqueeze(-2), min=0) * present / 8

    card = torch.cat([cards, shortfall(ME_BLOCK), shortfall(OPP_BLOCK)], -1)  # [..., 15, 40]
    qf = q.float()
    q_buy, q_res = qf[..., :CARD_DIM], qf[..., CARD_DIM : 2 * CARD_DIM]
    t_line, t_cell = qf[..., 2 * CARD_DIM : 2 * CARD_DIM + 7], qf[..., 2 * CARD_DIM + 7 :]
    buy = (card * q_buy.unsqueeze(-2)).sum(-1)  # pyramid 12, my reserved 3
    reserve = (card[..., :12, :] * q_res.unsqueeze(-2)).sum(-1)
    board = xf[..., :175].reshape(*lead, 25, 7)
    lines = (board * t_line.unsqueeze(-2)).sum(-1) @ M.t().float()  # M follows the model's dtype
    cells = (board * t_cell.unsqueeze(-2)).sum(-1)
    zeros = lambda k: xf.new_zeros(*lead, k)
    # action ids (engine/constants.ts): lines 0-144, cells 145-169, replenish 170, reserve 171-182 pyramid /
    # 183-185 deck, buy 186-197 pyramid / 198-200 reserved, then joker, steal, royal, discard, pass
    return torch.cat([lines, cells, zeros(1), reserve, zeros(3), buy, zeros(NUM_ACTIONS - 201)], -1)
