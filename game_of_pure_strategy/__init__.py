__version__ = "0.2.0"

import logging
import sys
from typing import Dict

import click

from .gops import (  # noqa: F401
    GameState,
    Strategy,
    optimize_player_strategy,
    get_strategies_for_possible_top_cards,
    get_optimal_game_strategy,
    visualize_strategies,
)


@click.command()
@click.option(
    "-n", "--n-cards", type=int, default=5, help="Number of cards in the deck"
)
@click.option(
    "-l",
    "--log-level",
    type=str,
    default="INFO",
    help="Log level: DEBUG/INFO/WARNING/ERROR",
)
def optimize(n_cards: int, log_level: str) -> Dict[int, Strategy]:
    logging.basicConfig(
        format="%(levelname)s: %(message)s", level=log_level, stream=sys.stderr
    )

    strategies = get_optimal_game_strategy(n_cards)
    print(visualize_strategies(n_cards, strategies))

    return strategies  # noqa: T484
