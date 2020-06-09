from typing import List, Optional, Iterable, Tuple, FrozenSet, Dict
from itertools import permutations, combinations, product
from dataclasses import dataclass

from game_of_pure_strategy.modeling import card_range


@dataclass(frozen=True)
class GameRound:
    player_move: int
    opponent_move: int
    deck_card: int


@dataclass(frozen=True)
class GameState:
    player_cards: FrozenSet[int]
    opponent_cards: FrozenSet[int]
    deck_cards: Tuple[int, ...]

    @classmethod
    def of_size(cls, size: int, max_card: int) -> Iterable["GameState"]:
        """
        Iterate over all possible states of the given size,
        where the card values are in range [1, max_card]
        """
        assert size >= 0, "Size has to be greater than zero"
        assert max_card >= 1, "Max card has to be greater than one"

        def _get_combinations():
            return combinations(card_range(max_card), size)

        def _get_permutations():
            return permutations(card_range(max_card), size)

        # TODO: Only generate combinations in-order (value cache is not enough to go beyond n=8)
        card_combinations = product(
            _get_combinations(), _get_combinations(), _get_permutations()
        )
        for player_cards, opponent_cards, deck_cards in card_combinations:
            game_state = cls(
                player_cards=frozenset(player_cards),
                opponent_cards=frozenset(opponent_cards),
                deck_cards=tuple(deck_cards)
            )
            yield game_state

    def __post_init__(self):
        """ Checks if all card collections have the same length. """
        same_number_of_cards = (
            len(self.player_cards) == len(self.opponent_cards) == len(self.deck_cards)
        )
        assert same_number_of_cards, "All stacks in GameState must have same number of cards!"

    def __len__(self) -> int:
        """ Get number of remaining cards (moves to play). """
        return len(self.player_cards)  # all card collections have the same length

    def opposite(self) -> "GameState":
        """ Get a state with player cards swapped with opponent cards. """
        opposite_player_cards = self.opponent_cards
        opposite_opponent_cards = self.player_cards
        opposite_game_state = GameState(
            opposite_player_cards,
            opposite_opponent_cards,
            self.deck_cards
        )
        return opposite_game_state

    def possible_moves(self) -> Iterable[Tuple[GameRound, "GameState"]]:
        """
        Iterate over possible moves in a given state.
        Each move is a tuple: (game round, game state after the round)
        """
        card_combinations = product(
            self.player_cards,
            self.opponent_cards,
            [self.deck_cards[0]]
        )
        for player_card, opponent_card, deck_card in card_combinations:
            game_round = GameRound(
                player_move=player_card,
                opponent_move=opponent_card,
                deck_card=deck_card
            )

            remaining_player_cards = self.player_cards - frozenset([player_card])
            remaining_opponent_cards = self.opponent_cards - frozenset([opponent_card])
            remaining_deck_cards = tuple(self.deck_cards[1:])
            next_game_state = GameState(
                player_cards=remaining_player_cards,
                opponent_cards=remaining_opponent_cards,
                deck_cards=remaining_deck_cards,
            )

            yield game_round, next_game_state


class ValueCache:
    def __init__(self):
        self.values = dict()

    def __getitem__(self, game_state: GameState) -> float:
        # if player and opponent have same cards, the expected value is always 0
        if game_state.player_cards == game_state.opponent_cards:
            return 0

        # firstly, try to find the value of the game in the cache
        value = self.values.get(game_state)
        if value is not None:
            return value

        # secondly, try to find the value of the opposite game in the cache and negate it
        opposite_state = game_state.opposite()
        opposite_value = self.values.get(opposite_state)
        if opposite_value is None:
            raise KeyError("None of {game state, opposite game state} are present in cache!")

        value = -opposite_value
        return value

    def get(self, game_state: GameState) -> Optional[float]:
        try:
            return self[game_state]
        except KeyError:
            return None

    def __setitem__(self, game_state: GameState, player_value: float):
        self.values[game_state] = player_value


def sign(i: int) -> int:
    if i > 0:
        return 1
    elif i == 0:
        return 0
    else:
        return -1


def move_value(game_round: GameRound, remaining_game_value: float) -> float:
    value = game_round.deck_card * sign(game_round.player_move - game_round.opponent_move)
    value += remaining_game_value
    return value


def compute_payoff_matrix(max_cards: int) -> Dict[Tuple[int, int], float]:
    # NOTE: assuming that card values are in range [1, max_cards] (inclusive on both sides)

    previous_value_cache = ValueCache()
    for n_cards in range(1, max_cards):
        # base case (n_cards = 0) is automatically handled by ValueCache

        current_value_cache = ValueCache()
        for game_state in GameState.of_size(n_cards, max_card=max_cards):
            if current_value_cache.get(game_state) is not None:
                continue  # value was already computed (eg. extracted from opposite state)

            sum_values = 0
            n_rounds = 0
            for game_round, next_game_state in game_state.possible_moves():
                remaining_game_value = previous_value_cache[next_game_state]
                sum_values += move_value(game_round, remaining_game_value)
                n_rounds += 1

            expected_game_value = sum_values / n_rounds
            current_value_cache[game_state] = expected_game_value

        # print(f"n_cards: {n_cards}")
        # print(previous_value_cache.values)
        # print(current_value_cache.values)
        # print("")
        previous_value_cache = current_value_cache

    # there are N! starting states (each state corresponds to different deck permutation)
    matrix = dict()  # (player move, opponent move) => [move values for possible starting states]
    for starting_state in GameState.of_size(max_cards, max_card=max_cards):
        for game_round, next_game_state in starting_state.possible_moves():
            remaining_game_value = previous_value_cache[next_game_state]
            game_value = move_value(game_round, remaining_game_value)

            key = (game_round.player_move, game_round.opponent_move)
            try:
                matrix[key].append(game_value)
            except KeyError:
                matrix[key] = [game_value]

    # expected move value = average move value across possible starting states
    for move in matrix:
        values_for_fixed_deck = matrix[move]
        matrix[move] = sum(values_for_fixed_deck) / len(values_for_fixed_deck)

    return matrix
