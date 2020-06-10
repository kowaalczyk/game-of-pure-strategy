from typing import Iterable, Dict, Tuple

from tabulate import tabulate


PayoffMatrix = Dict[Tuple[int, int], float]


def card_range(max_card_value: int) -> Iterable[int]:
    return range(1, max_card_value + 1)


def visualize_matrix(n_cards: int, matrix: PayoffMatrix) -> str:
    header = ["player \ opponent"] + list(card_range(n_cards))

    table = []
    for player_card in card_range(n_cards):
        values = [
            matrix[(player_card, opponent_card)] for opponent_card in card_range(n_cards)
        ]
        row = [player_card] + values
        table.append(row)

    return tabulate(table, headers=header, tablefmt="github", floatfmt=".4f")
