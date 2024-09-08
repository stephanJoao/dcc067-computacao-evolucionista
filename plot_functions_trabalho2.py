import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from opfunu.cec_based.cec2014 import F52014, F92014


def read_solutions(function):
    os.chdir("results_2D")
    f_name = function.__self__.__class__.__name__
    pop_size = 50
    elite_best = 0.2
    elite_worst = 0.2

    solutions = []
    df = pd.read_csv(
        # f"{f_name}_{pop_size}_{elite_best}_{elite_worst}_GA.csv"
        f"{f_name}_{pop_size}_{elite_best}_{elite_worst}_ABC.csv"
    )
    s = np.array(
        [
            np.fromstring(i[1:-1], sep=" ")
            for i in df["best_solution"].values
        ]
    ).mean(axis=0)
    solutions.append(s)
    solutions = np.array(solutions)
    os.chdir("..")

    return solutions


def plot_functions_2d(function, plot_solutions=True):
    x = np.linspace(-100, 100, 500)
    y = np.linspace(-100, 100, 500)

    X, Y = np.meshgrid(x, y)
    Z = np.array(
        [
            function(np.array([xx, yy]))
            for xx, yy in zip(np.ravel(X), np.ravel(Y))
        ]
    ).reshape(X.shape)

    plt.figure()
    plt.contourf(
        X,
        Y,
        Z,
        cmap="viridis",
        levels=np.linspace(np.min(Z), np.max(Z), 100),
    )
    if plot_solutions:
        solutions = read_solutions(function)
        for solution in solutions:
            plt.plot(solution[0], solution[1], "ro")
    plt.title(f"Contour plot of {function.__self__.__class__.__name__}")

    plt.colorbar()
    plt.tight_layout()
    plt.savefig(
        f"plots/contour_plot_{function.__self__.__class__.__name__}.png",
        dpi=300,
    )


def plot_functions_3d(function):
    x = np.linspace(-100, 100, 500)
    y = np.linspace(-100, 100, 500)

    X, Y = np.meshgrid(x, y)
    Z = np.array(
        [
            function(np.array([xx, yy]))
            for xx, yy in zip(np.ravel(X), np.ravel(Y))
        ]
    ).reshape(X.shape)

    plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis")
    plt.title(f"Surface plot of {function.__self__.__class__.__name__}")

    plt.xlim(-100, 100)
    plt.ylim(-100, 100)
    ax.set_zlim(np.min(Z), np.max(Z))
    plt.tight_layout()
    plt.savefig(
        f"plots/surface_plot_{function.__self__.__class__.__name__}.png",
        dpi=300,
    )


if __name__ == "__main__":
    ndim = 2
    f5 = F52014(ndim=ndim).evaluate
    f9 = F92014(ndim=ndim).evaluate

    # if plots dir does not exist, create it
    if not os.path.exists("plots"):
        os.makedirs("plots")

    print("Plotting 2D")
    plot_functions_2d(f5)
    plot_functions_2d(f9)
    print("Plotting 3D")
    plot_functions_3d(f5)
    plot_functions_3d(f9)
