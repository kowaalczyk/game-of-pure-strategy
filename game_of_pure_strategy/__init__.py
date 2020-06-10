__version__ = "0.1.0"

import logging
import sys

import click

from .gops import (
    GameState,
    Strategy,
    optimize_player_strategy,
    get_strategies_for_possible_top_cards,
    get_optimal_strategy,
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
def optimize(n_cards: int, log_level: str):
    logging.basicConfig(
        format="%(levelname)s: %(message)s", level=log_level, stream=sys.stderr
    )

    strategies = get_optimal_strategy(n_cards)
    print(visualize_strategies(n_cards, strategies))

    return strategies
