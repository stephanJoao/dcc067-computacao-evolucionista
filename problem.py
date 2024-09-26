import numpy as np
from mealpy.utils.problem import FloatVar, IntegerVar

D = 160
d = 90
B_w = 30
r_i = r_o = 11.033


def g1(phi_0, Z, D_b, D_m):
    return (phi_0 / (2 * np.arcsin(D_b / D_m))) - Z + 1 <= 0


def g2(D_b, K_Dmin, D, d):
    return 2 * D_b - K_Dmin * (D - d) > 0


def g3(K_Dmax, D, d, D_b):
    return K_Dmax * (D - d) - 2 * D_b >= 0


def g4(zeta, Bw, D_b):
    return zeta * Bw - D_b <= 0


def g5(D_m, D, d):
    return D_m - 0.5 * (D + d) >= 0


def g6(epsilon, D, d, D_m):
    return (0.5 + epsilon) * (D + d) - D_m >= 0


def g7(D, D_m, D_b, Epsilon):
    return 0.5 * (D - D_m - D_b) - Epsilon * D_b >= 0


def g8(f_i):
    return f_i - 0.515 >= 0


def g9(f_o):
    return f_o - 0.515 >= 0


def f_c(gamma, f_i, f_o):
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


def evaluate(solution):
    D_m, D_b, Z, f_i, f_o, K_D_min, K_D_max, Epsilon, epsilon, zeta = solution

    T = D - d - 2 * D_b
    x = (
        ((D - d) / 2 - 3 * (T / 4)) ** 2
        + (D / 2 - T / 4 - D_b) ** 2
        - (d / 2 + T / 4) ** 2
    )
    y = 2 * ((D - d) / 2 - 3 * (T / 4)) * (D / 2 - T / 4 - D_b)
    print(x / y)
    phi_0 = 2 * np.pi - np.arccos(x / y)
    gamma = D_b / D_m
    f_i = r_i / D_b
    f_o = r_o / D_b

    if D_b <= 25.4:
        f = f_c(gamma, f_i, f_o) * Z ** (2 / 3) * D_b ** (1.8)
    else:
        f = 3.647 * f_c(gamma, f_i, f_o) * Z ** (2 / 3) * D_b ** (1.4)

    penalty = 0
    increase = 0

    if not g1(phi_0, Z, D_b, D_m):
        penalty += increase
        print("g1")
    if not g2(D_b, K_D_min, D, d):
        penalty += increase
        print("g2")
    if not g3(K_D_max, D, d, D_b):
        penalty += increase
        print("g3")
    if not g4(zeta, B_w, D_b):
        penalty += increase
        print("g4")
    if not g5(D_m, D, d):
        penalty += increase
        print("g5")
    if not g6(epsilon, D, d, D_m):
        penalty += increase
        print("g6")
    if not g7(D, D_m, D_b, Epsilon):
        penalty += increase
        print("g7")
    if not g8(f_i):
        penalty += increase
        print("g8")
    if not g9(f_o):
        penalty += increase
        print("g9")

    return f + penalty


D_m_bounds = FloatVar(0.5 * (D + d), 0.6 * (D + d))
D_b_bounds = FloatVar(0.15 * (D - d), 0.45 * (D - d))
Z_bounds = IntegerVar(4, 50)
f_i_bounds = FloatVar(0.515, 0.6)
f_o_bounds = FloatVar(0.515, 0.6)
K_D_min_bounds = FloatVar(0.4, 0.5)
K_D_max_bounds = FloatVar(0.6, 0.7)
Epsilon_bounds = FloatVar(0.3, 0.4)
epsilon_bounds = FloatVar(0.02, 0.1)
zeta_bounds = FloatVar(0.6, 0.85)

bounds = [
    D_m_bounds,
    D_b_bounds,
    Z_bounds,
    f_i_bounds,
    f_o_bounds,
    K_D_min_bounds,
    K_D_max_bounds,
    Epsilon_bounds,
    epsilon_bounds,
    zeta_bounds,
]

if __name__ == "__main__":
    solution = [
        125.7171,
        21.423,
        11,
        0.515,
        0.515,
        0.4159,
        0.651,
        0.300043,
        0.0223,
        0.751,
    ]

    print(evaluate(solution))
