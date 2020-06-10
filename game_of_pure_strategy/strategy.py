import mip

from game_of_pure_strategy.modeling import card_range, PayoffMatrix


def find_nash_equilibrium(max_card: int, payoff_matrix: PayoffMatrix):
    player_lp = optimize_player_strategy(max_card, payoff_matrix)
    opponent_lp = optimize_opponent_strategy(max_card, payoff_matrix)

    assert player_lp.var_by_name("v").x == opponent_lp.var_by_name("v").x


def optimize_player_strategy(max_card: int, payoff_matrix: PayoffMatrix) -> mip.Model:
    """ Get a solved linear program that optimizes player strategy """

    lp = mip.Model("player_strategy", solver_name=mip.CBC)
    x = [lp.add_var(f"x_{i}", var_type=mip.CONTINUOUS) for i in card_range(max_card)]
    v = lp.add_var("v", var_type=mip.CONTINUOUS, lb=-mip.INF)

    for opponent_card in card_range(max_card):
        transposed_row = [
            payoff_matrix[(opponent_card, player_card)] for player_card in card_range(max_card)
        ]
        constraint = mip.xsum(transposed_row[i] * x[i] for i in range(max_card)) - v >= 0
        lp += constraint, f"strategy_against_{opponent_card}"

    lp += mip.xsum(x[i] for i in range(max_card)) == 1, "probability_distribution"
    lp.objective = mip.maximize(v)

    lp.optimize(max_seconds=30)
    return lp


def optimize_opponent_strategy(max_card: int, payoff_matrix: PayoffMatrix) -> mip.Model:
    """ Get a solved linear program that optimizes opponent strategy """

    lp = mip.Model("player_strategy", solver_name=mip.CBC)
    y = [lp.add_var(f"y_{i}", var_type=mip.CONTINUOUS) for i in card_range(max_card)]
    v = lp.add_var("v", var_type=mip.CONTINUOUS, lb=-mip.INF)

    for player_card in card_range(max_card):
        row = [
            payoff_matrix[(player_card, opponent_card)] for opponent_card in card_range(max_card)
        ]
        constraint = mip.xsum(row[i] * y[i] for i in range(max_card)) - v <= 0
        lp += constraint, f"strategy_against_{player_card}"

    lp += mip.xsum(y[i] for i in range(max_card)) == 1, "probability_distribution"
    lp.objective = mip.minimize(v)

    lp.optimize(max_seconds=30)  # todo: parametrize optimizer
    return lp
