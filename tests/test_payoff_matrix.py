import pytest

from game_of_pure_strategy.payoff_matrix import compute_payoff_matrix

TESTS = [
    (1, {(1, 1): 0.0}),
    (2, {(1, 1): 0.0, (1, 2): 0.0, (2, 1): 0.0, (2, 2): 0.0}),
    (3, {
        (1, 1): 0.0,
        (1, 2): -1.0,
        (1, 3): 1.0,
        (2, 1): 1.0,
        (2, 2): 0.0,
        (2, 3): -1.0,
        (3, 1): -1.0,
        (3, 2): 1.0,
        (3, 3): 0.0,
    }),
]


@pytest.mark.parametrize("test_case", TESTS)
def test_payoff_matrix(test_case):
    n_cards, expected_result = test_case

    result = compute_payoff_matrix(n_cards)

    assert result == expected_result
