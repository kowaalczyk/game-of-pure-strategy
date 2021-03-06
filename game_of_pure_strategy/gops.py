from dataclasses import dataclass
from typing import FrozenSet, Dict, Iterable, Iterator, Tuple, List
from itertools import combinations_with_replacement, combinations, product
from functools import wraps
from time import time
import logging

import mip
from tabulate import tabulate


def logged(func):  # noqa: T484
    @wraps(func)
    def inner(*args, **kwargs):  # noqa: T484
        logging.debug(func.__name__)
        return func(*args, **kwargs)

    return inner


def card_range(max_card: int) -> Iterable[int]:
    return range(1, max_card + 1)


@dataclass(frozen=True)
class GameState:
    """
    Represents a state of the game, frozen sets are used for storing collections,
    making game state hashable, which in turn allows using it in dicts' keys.
    This improves readability, since we can express mapping like `value[state] = 1.0`.
    """

    player_cards: FrozenSet[int]
    opponent_cards: FrozenSet[int]
    deck_cards: FrozenSet[int]

    @staticmethod
    def of_size(size: int, max_card: int) -> Iterator["GameState"]:
        """
        Iterate over games of the given size and cards in range [1, max_card],
        where player cards are equal or worse than opponent cards (values of
        opposite games can be calculated by symmetry: V(P, O, D) = -V(O, P, D),
        so we get 2x speed up by not iterating over all possible games).
        """
        possible_player_cards = combinations_with_replacement(
            combinations(card_range(max_card), size), 2
        )
        possible_decks = combinations(card_range(max_card), size)
        possible_games = product(possible_player_cards, possible_decks)
        for (player_cards, opponent_cards), deck_cards in possible_games:
            possible_game = GameState(
                player_cards=frozenset(player_cards),
                opponent_cards=frozenset(opponent_cards),
                deck_cards=frozenset(deck_cards),
            )
            yield possible_game

    @classmethod
    def empty(cls) -> "GameState":
        empty_game = GameState(
            player_cards=frozenset(),
            opponent_cards=frozenset(),
            deck_cards=frozenset(),
        )
        return empty_game

    def after_round(
        self, revealed_card: int, player_move: int, opponent_move: int
    ) -> "GameState":
        """
        Get a state created from the current state when:
        - the `revealed_card` is revealed and removed from the deck
        - player played `player_move` card
        - opponent played `opponent_move` card
        """

        remaining_deck = self.deck_cards - frozenset([revealed_card])
        remaining_player = self.player_cards - frozenset([player_move])
        remaining_opponent = self.opponent_cards - frozenset([opponent_move])

        game_state_after_round = GameState(
            player_cards=remaining_player,
            opponent_cards=remaining_opponent,
            deck_cards=remaining_deck,
        )
        return game_state_after_round

    def opposite(self) -> "GameState":
        """ Get a state with player's and opponent's cards swapped """
        opposite_game_state = GameState(
            player_cards=self.opponent_cards,
            opponent_cards=self.player_cards,
            deck_cards=self.deck_cards,
        )
        # TODO: For some reason uncommenting this breaks results, re-visit later:
        return opposite_game_state


@dataclass(frozen=True)
class Strategy:
    card_probabilities: Dict[int, float]
    expected_value: float


Matrix = Dict[Tuple[int, int], float]


def sign(i: int) -> float:
    if i > 0:
        return 1.0
    elif i == 0:
        return 0.0
    else:
        return -1.0


@logged
def optimize_player_strategy(
    player_cards: List[int], opponent_cards: List[int], payoff_matrix: Matrix
) -> Strategy:
    """
    Get the optimal strategy for the player, by solving
    a simple linear program based on payoff matrix.
    """
    lp = mip.Model("player_strategy", solver_name=mip.CBC)
    lp.verbose = False  # the problems are simple and we don't need to see the output

    x = [lp.add_var(f"x_{card}", var_type=mip.CONTINUOUS) for card in player_cards]
    v = lp.add_var("v", var_type=mip.CONTINUOUS, lb=-mip.INF)

    for opponent_card in opponent_cards:
        transposed_row = [
            payoff_matrix[(player_card, opponent_card)] for player_card in player_cards
        ]
        constraint = (
            mip.xsum(transposed_row[i] * x_i for i, x_i in enumerate(x)) - v >= 0
        )
        lp += constraint, f"strategy_against_{opponent_card}"
        logging.debug(f"constraint={constraint}")

    lp += mip.xsum(x) == 1, "probability_distribution"
    lp.objective = mip.maximize(v)

    # all variables are continuous so we only need to solve relaxed problem
    lp.optimize(max_seconds=30, relax=True)
    if lp.status is not mip.OptimizationStatus.OPTIMAL:
        logging.error(f"lp.status={lp.status}")
        raise RuntimeError(
            f"Solver couldn't optimize the problem and returned status {lp.status}"
        )

    strategy = Strategy(
        card_probabilities={
            card: lp.var_by_name(f"x_{card}").x for card in player_cards
        },
        expected_value=lp.var_by_name("v").x,
    )
    logging.debug(f"strategy.expected_value={strategy.expected_value}")
    logging.debug("\n")
    return strategy


@logged
def get_strategies_for_possible_top_cards(
    game_state: GameState, cached_game_values: Dict[GameState, float]
) -> Dict[int, Strategy]:
    """ Get a player strategy for every possible card from the deck. """
    if len(game_state.deck_cards) == 0:
        return dict()

    if len(game_state.deck_cards) == 1:

        def get_item(frozen_set: FrozenSet[int]) -> int:
            return next(iter(frozen_set))

        player_move = get_item(game_state.player_cards)
        opponent_move = get_item(game_state.opponent_cards)
        top_card = get_item(game_state.deck_cards)

        move_value = top_card * sign(player_move - opponent_move)  # noqa: T484
        only_possible_strategy = Strategy(
            card_probabilities={player_move: 1.0}, expected_value=move_value
        )
        return {top_card: only_possible_strategy}

    strategies: Dict[int, Strategy] = dict()
    for top_card in game_state.deck_cards:
        payoff_matrix: Matrix = dict()
        possible_moves = product(game_state.player_cards, game_state.opponent_cards)
        for player_move, opponent_move in possible_moves:
            if player_move == opponent_move:
                move_value = 0.0  # noqa: T484
            else:
                move_value = top_card * sign(player_move - opponent_move)  # noqa: T484

            # either continuation or opposite of continuation is present in cached_game_values
            continuation = game_state.after_round(top_card, player_move, opponent_move)
            continuation_value = cached_game_values.get(continuation)
            if continuation_value is None:
                continuation_value = -cached_game_values[continuation.opposite()]

            payoff_matrix[(player_move, opponent_move)] = (
                move_value + continuation_value
            )

        logging.debug(f"game_state={game_state}")
        logging.debug(f"top_card={top_card}")
        logging.debug(f"payoff_matrix={payoff_matrix}")
        logging.debug("")
        strategy = optimize_player_strategy(
            list(game_state.player_cards),
            list(game_state.opponent_cards),
            payoff_matrix,
        )
        strategies[top_card] = strategy

    return strategies


@logged
def get_optimal_game_strategy(game_size: int) -> Dict[int, Strategy]:
    """
    Completes optimal strategy for an entire game, using a dynamic programming
    approach to incrementally solve larger games (from 1 to `game_size`).

    Returns optimal strategy for every possible top card in [1, game_size].
    """

    cached_game_values: Dict[GameState, float] = {GameState.empty(): 0.0}
    for n_cards in card_range(game_size - 1):
        logging.info(f"Calculating strategies for n_cards={n_cards}")
        logging.debug(f"cached_game_values={cached_game_values}")
        start = time()

        game_values: Dict[GameState, float] = dict()
        for game_state in GameState.of_size(n_cards, max_card=game_size):
            strategies = get_strategies_for_possible_top_cards(
                game_state, cached_game_values
            )
            strategy_values = [
                strategy.expected_value for strategy in strategies.values()
            ]
            game_value = sum(strategy_values) / len(strategies)
            game_values[game_state] = game_value

        end = time()
        logging.info(f"Calculated strategies for n_cards={n_cards} in {end-start:.2f}s")
        cached_game_values = game_values

    logging.info(f"Calculating strategies for n_cards={game_size}")
    logging.debug(f"cached_game_values={cached_game_values}")
    game_state = next(
        GameState.of_size(game_size, max_card=game_size)
    )  # there is only one such game
    strategies = get_strategies_for_possible_top_cards(game_state, cached_game_values)
    return strategies  # noqa: T484


def visualize_strategies(n_cards: int, strategies: Dict[int, Strategy]) -> str:
    """ Creates a printable representation of strategy dict. """
    header = ["top card \\ player move"] + list(card_range(n_cards))  # noqa: T484

    table = []
    for top_card in card_range(n_cards):
        strategy = strategies[top_card]
        values = [strategy.card_probabilities[card] for card in card_range(n_cards)]
        row = [top_card] + values  # noqa: T484
        table.append(row)

    return tabulate(table, headers=header, tablefmt="github", floatfmt=".4f")
