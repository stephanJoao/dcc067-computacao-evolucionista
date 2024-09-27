import numpy as np
from mealpy import FloatVar, GA, CEM, IntegerVar, ABC
import matplotlib.pyplot as plt


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
    return 2 * np.pi - 2 * np.arccos(x(D_b) / y(D_b))


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

def g10(solution):
    _, D_b, _, f_i, _, _, _, _, _, _ = solution
    result = round(f_i - r_i/D_b, 3)
    return abs(result)

def g11(solution):
    _, D_b, _, _, f_o, _, _, _, _, _ = solution
    result = round(f_o - r_o/D_b, 3)
    return abs(result)

def evaluate_with_penalty(solution):
    _, D_b, Z, _, _, _, _, _, _, _ = solution

    coef = 1 if D <= 254 else 3.647
    exp = 1.8 if D <= 254 else 1.4

    val = coef * f_c(solution) * Z ** (2 / 3) * D_b**exp

    weights = [100000] * 11

    # c = g1(solution) + g2(solution) + g3(solution) + g4(solution) + g5(solution) + g6(solution) + g7(solution) + g8(solution) + g9(solution) + g10(solution) + g11(solution)
    # if (c > 0):
    #     return 0

    g1_v = 1 if g1(solution) > 0 else 0 
    g2_v = 1 if g2(solution) > 0 else 0
    g3_v = 1 if g3(solution) > 0 else 0
    g4_v = 1 if g4(solution) > 0 else 0
    g5_v = 1 if g5(solution) > 0 else 0
    g6_v = 1 if g6(solution) > 0 else 0
    g7_v = 1 if g7(solution) > 0 else 0
    g8_v = 1 if g8(solution) > 0 else 0
    g9_v = 1 if g9(solution) > 0 else 0
    g10_v = 1 if g10(solution) > 0 else 0
    g11_v = 1 if g11(solution) > 0 else 0

    penalty = val / 11

    result = val - g1_v * penalty - g2_v * penalty - g3_v * penalty - g4_v * penalty - g5_v * penalty - g6_v * penalty - g7_v * penalty - g8_v * penalty - g9_v * penalty - g10_v * penalty - g11_v * penalty

    # result = (
    #     val
    #     - g1(solution) * weights[0]
    #     - g2(solution) * weights[1]
    #     - g3(solution) * weights[2]
    #     - g4(solution) * weights[3]
    #     - g5(solution) * weights[4]
    #     - g6(solution) * weights[5]
    #     - g7(solution) * weights[6]
    #     - g8(solution) * weights[7]
    #     - g9(solution) * weights[8]
    #     - g10(solution) * weights[9]
    #     - g11(solution) * weights[10]
    # )
    return result


def check_bounds(solution, bounds):
    for i, bound in enumerate(bounds):
        if not bound.lb <= solution[i] <= bound.ub:
            return False
    return True


def evaluate(solution):
    _, D_b, Z, _, _, _, _, _, _, _ = solution

    coef = 1 if D <= 254 else 3.647
    exp = 1.8 if D <= 254 else 1.4

    val = coef * f_c(solution) * Z ** (2 / 3) * D_b**exp

    return val


def print_violated_constraints(solution):
    print("g1:", g1(solution))
    print("g2:", g2(solution))
    print("g3:", g3(solution))
    print("g4:", g4(solution))
    print("g5:", g5(solution))
    print("g6:", g6(solution))
    print("g7:", g7(solution))
    print("g8:", g8(solution))
    print("g9:", g9(solution))
    print("g10:", g10(solution))
    print("g11:", g11(solution))


if __name__ == "__main__":

    bounds = [
        FloatVar(0.5 * (D + d), 0.6 * (D + d)),  # D_m
        FloatVar(0.15 * (D - d), 0.45 * (D - d)),  # D_b
        FloatVar(4.0, 50.0),  # Z
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
        "log_to": "None",
        "save_population": True
    }

    # model = GA.EliteSingleGA(
    #     epoch=100,
    #     pop_size=50,
    #     elite_best=0.1,
    #     elite_worst=0.3,
    #     selection="tournament",
    #     mutation="swap",
    #     crossover="arithmetic",
    # )

    model = ABC.OriginalABC(
        epoch=100,
        pop_size=50, 
        n_limits=25
    )

    data = []
    average = 0
    for i in range(10):
        best = model.solve(problem_dict)

        # print(best.solution)
        print(best.target.fitness)
        print_violated_constraints(best.solution)

        data.append(model.history.list_global_best_fit)
        average += best.target.fitness

    # Plot all lines in the same plot
    for i in range(10):
        plt.plot(data[i])

    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.title("Fitness x Iteration")
    plt.show()

    print("Average:", average / 10)
    