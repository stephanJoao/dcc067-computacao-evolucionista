import numpy as np
from mealpy import FloatVar, GA, ABC

D = 160
d = 90
B_w = 30
r_i = r_o = 11.033


def T(D_b):
    return D - d - 2 * D_b


def x(D_b):
    return (
        ((D - d) / 2 - 3 * (T(D_b) / 4)) ** 2
        + (D / 2 - T(D_b) / 4 - D_b) ** 2
        - (d / 2 + T(D_b) / 4) ** 2
    )


def y(D_b):
    return 2 * ((D - d) / 2 - 3 * (T(D_b) / 4)) * (D / 2 - T(D_b) / 4 - D_b)


def phi_0(D_b):
    return 2 * np.pi - np.arccos(x(D_b) / y(D_b))


def f_c(solution):
    D_m, D_b, _, f_i, f_o, _, _, _, _, _ = solution
    gamma = D_b / D_m
    return (
        37.91
        * (
            1
            + (
                1.04
                * ((1 - gamma) / (1 + gamma)) ** 1.72
                * ((f_i * (2 * f_o - 1)) / (f_o * (2 * f_i - 1))) ** 0.41
            )
            ** (10 / 3)
        )
        ** (-0.3)
        * ((gamma**0.3 * (1 - gamma) ** 1.39) / (1 + gamma) ** (1 / 3))
        * ((2 * f_i) / (2 * f_i - 1)) ** (0.41)
    )


def g1(solution):
    D_m, D_b, Z, _, _, _, _, _, _, _ = solution
    result = phi_0(D_b) / (2 * np.arcsin(D_b / D_m)) - Z + 1
    return result if result > 0 else 0


def g2(solution):
    _, D_b, _, _, _, K_D_min, _, _, _, _ = solution
    result = 2 * D_b - K_D_min * (D - d)
    return abs(result) if result <= 0 else 0


def g3(solution):
    _, D_b, _, _, _, _, K_D_max, _, _, _ = solution
    result = K_D_max * (D - d) - 2 * D_b
    return abs(result) if result < 0 else 0


def g4(solution):
    _, D_b, _, _, _, _, _, _, _, zeta = solution
    result = zeta * B_w - D_b
    return result if result > 0 else 0


def g5(solution):
    D_m, _, _, _, _, _, _, _, _, _ = solution
    result = D_m - 0.5 * (D + d)
    return abs(result) if result < 0 else 0


def g6(solution):
    D_m, _, _, _, _, _, _, _, epsilon, _ = solution
    result = (0.5 + epsilon) * (D + d) - D_m
    return abs(result) if result < 0 else 0


def g7(solution):
    D_m, D_b, _, _, _, _, _, Epsilon, _, _ = solution
    result = 0.5 * (D - D_m - D_b) - Epsilon * D_b
    return abs(result) if result < 0 else 0


def g8(solution):
    _, _, _, f_i, _, _, _, _, _, _ = solution
    result = f_i - 0.515
    return abs(result) if result < 0 else 0


def g9(solution):
    _, _, _, _, f_o, _, _, _, _, _ = solution
    result = f_o - 0.515
    return abs(result) if result < 0 else 0



def evaluate_with_penalty(solution):
    _, D_b, Z, _, _, _, _, _, _, _ = solution

    coef = 1 if D <= 25.4 else 3.647
    exp = 1.8 if D_b <= 25.4 else 1.4

    val = coef * f_c(solution) * Z ** (2 / 3) * D_b**exp

    weights = [1000000] * 9

    return (
        val
        - g1(solution) * weights[0]
        - g2(solution) * weights[1]
        - g3(solution) * weights[2]
        - g4(solution) * weights[3]
        - g5(solution) * weights[4]
        - g6(solution) * weights[5]
        - g7(solution) * weights[6]
        - g8(solution) * weights[7]
        - g9(solution) * weights[8]
    )


def evaluate(solution):
    _, D_b, Z, _, _, _, _, _, _, _ = solution

    coef = 1 if D <= 25.4 else 3.647
    exp = 1.8 if D_b <= 25.4 else 1.4

    val = coef * f_c(solution) * Z ** (2 / 3) * D_b**exp

    return val


if __name__ == "__main__":

    bounds = [
        FloatVar(0.5 * (D + d), 0.6 * (D + d)),  # D_m
        FloatVar(0.15 * (D - d), 0.45 * (D - d)),  # D_b
        FloatVar(4, 50),  # Z
        FloatVar(0.515, 0.6),  # f_i
        FloatVar(0.515, 0.6),  # f_o
        FloatVar(0.4, 0.5),  # K_D_min
        FloatVar(0.6, 0.7),  # K_D_max
        FloatVar(0.3, 0.4),  # Epsilon
        FloatVar(0.02, 0.1),  # epsilon
        FloatVar(0.6, 0.85),  # zeta
    ]

    problem_dict = {
        "obj_func": evaluate_with_penalty,
        "bounds": bounds,
        "minmax": "max",
    }

    model = GA.EliteSingleGA(
        epoch=1000,
        n_pop=200,
        selection="tournament",
        crossover="uniform"
    )
    # model = ABC.OriginalABC(epoch=1000)

    best = model.solve(problem_dict)

    print(best.target.fitness)
    print(best.solution)

    result = evaluate(best.solution)
    print(result)
