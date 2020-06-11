import pytest

from game_of_pure_strategy import (
    __version__,
    GameState,
    Strategy,
    optimize_player_strategy,
    get_strategies_for_possible_top_cards,
    get_optimal_game_strategy,
)


# flake8: noqa


def test_version():
    assert __version__ == "0.1.0"


def test_opposite_state():
    game_state = GameState(
        player_cards=frozenset([1]),
        opponent_cards=frozenset([2]),
        deck_cards=frozenset([3]),
    )

    double_opposite_state = game_state.opposite().opposite()

    assert double_opposite_state == game_state


def test_state_after_round():
    game_state = GameState(
        player_cards=frozenset([1]),
        opponent_cards=frozenset([2]),
        deck_cards=frozenset([3]),
    )

    next_state = game_state.after_round(3, 1, 2)

    assert next_state == GameState.empty()


def assert_strategy_equal(strategy, expected_strategy):
    assert strategy.expected_value == pytest.approx(expected_strategy.expected_value)
    for card in expected_strategy.card_probabilities:
        assert strategy.card_probabilities[card] == pytest.approx(
            expected_strategy.card_probabilities[card]
        )


OPTIMIZATION_TESTS = [
    (
        # deck: {1,2}, top: 1
        [1, 2],
        [1, 2],
        {(1, 1): 0.0, (1, 2): 1.0, (2, 1): -1.0, (2, 2): 0.0},
        Strategy(card_probabilities={1: 1.0, 2: 0.0}, expected_value=0.0),
    ),
    (
        # deck: {1, 2}, top: 2
        [1, 2],
        [1, 2],
        {(1, 1): 0.0, (1, 2): -1.0, (2, 1): 1.0, (2, 2): 0.0},
        Strategy(card_probabilities={1: 0.0, 2: 1.0}, expected_value=0.0),
    ),
    (
        # deck: {1, 2}, top: 2
        [1, 2],
        [2, 3],
        {(1, 2): -3.0, (1, 3): -2.0, (2, 2): -1.0, (2, 3): -3.0},
        Strategy(
            card_probabilities={1: 0.6666666, 2: 0.3333333}, expected_value=-2.3333333
        ),
    ),
    (
        # deck: {1, 2}, top: 2
        [2, 3],
        [1, 2],
        {(2, 1): 3.0, (2, 2): 1.0, (3, 1): 2.0, (3, 2): 3.0},
        Strategy(
            card_probabilities={2: 0.3333333, 3: 0.6666666}, expected_value=2.3333333
        ),
    ),
]


@pytest.mark.parametrize("test_case", OPTIMIZATION_TESTS)
def test_optimization(test_case):
    player_cards, opponent_cards, payoff_matrix, expected_strategy = test_case

    strategy = optimize_player_strategy(player_cards, opponent_cards, payoff_matrix)

    assert_strategy_equal(strategy, expected_strategy)


def test_recursion():
    pass


E2E_TESTS = [
    (1, {1: Strategy(card_probabilities={1: 1.0}, expected_value=0.0)}),
    (
        2,
        {
            1: Strategy(card_probabilities={1: 1.0, 2: 0.0}, expected_value=0.0),
            2: Strategy(card_probabilities={1: 0.0, 2: 1.0}, expected_value=0.0),
        },
    ),
    (
        3,
        {
            1: Strategy(
                card_probabilities={1: 1.0, 2: 0.0, 3: 0.0}, expected_value=0.0
            ),
            2: Strategy(
                card_probabilities={1: 0.0, 2: 1.0, 3: 0.0}, expected_value=0.0
            ),
            3: Strategy(
                card_probabilities={1: 0.0, 2: 0.0, 3: 1.0}, expected_value=0.0
            ),
        },
    ),
]


@pytest.mark.parametrize("test_case", E2E_TESTS)
def test_e2e(test_case):
    n_cards, expected_result = test_case

    result = get_optimal_game_strategy(n_cards)

    for i in range(1, n_cards + 1):
        assert_strategy_equal(result[i], expected_result[i])
