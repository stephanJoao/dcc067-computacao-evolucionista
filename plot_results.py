import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_best(function, pop_size):
    os.chdir(f"results_{ndim}D")
    X = np.array(elite_best_values)
    Y = np.array(elite_worst_values)
    Z = np.zeros((len(X), len(Y)))
    for elite_best in elite_best_values:
        for elite_worst in elite_worst_values:
            df = pd.read_csv(
                f"{function}_{pop_size}_{elite_best}_{elite_worst}.csv"
            )
            Z[
                elite_best_values.index(elite_best),
                elite_worst_values.index(elite_worst),
            ] = df["best_fit"].mean()
    X, Y = np.meshgrid(X, Y)

    plt.figure()
    plt.imshow(Z, cmap="viridis", interpolation="nearest")
    plt.colorbar()
    plt.title(f"Mean of best fitness values ({function}, pop_size={pop_size})")
    plt.xlabel("Elite best")
    plt.ylabel("Elite worst")
    plt.xticks(np.arange(len(elite_best_values)), elite_best_values)
    plt.yticks(np.arange(len(elite_worst_values)), elite_worst_values)
    plt.tight_layout()
    plt.show()


def plot_line(function, value="list_fitness"):
    for pop_size in pop_sizes:
        # find best and worst elite_best and elite_worst values
        Z = np.zeros((len(elite_best_values), len(elite_worst_values)))
        for elite_best in elite_best_values:
            for elite_worst in elite_worst_values:
                df = pd.read_csv(
                    f"{function}_{pop_size}_{elite_best}_{elite_worst}.csv"
                )
                Z[
                    elite_best_values.index(elite_best),
                    elite_worst_values.index(elite_worst),
                ] = df["best_fit"].mean()
        best_elite_best, best_elite_worst = np.unravel_index(
            Z.argmin(), Z.shape
        )
        worst_elite_best, worst_elite_worst = np.unravel_index(
            Z.argmax(), Z.shape
        )

        plt.figure()
        for elite_best in elite_best_values:
            for elite_worst in elite_worst_values:
                df = pd.read_csv(
                    f"{function}_{pop_size}_{elite_best}_{elite_worst}.csv"
                )
                values = np.array(
                    [np.fromstring(i[1:-1], sep=" ") for i in df[value].values]
                ).mean(axis=0)
                if (
                    elite_best_values[best_elite_best] == elite_best
                    and elite_worst_values[best_elite_worst] == elite_worst
                ):
                    plt.plot(
                        values,
                        label=f"{pop_size}_{elite_best}_{elite_worst} (best)",
                        color="green",
                    )
                elif (
                    elite_best_values[worst_elite_best] == elite_best
                    and elite_worst_values[worst_elite_worst] == elite_worst
                ):
                    plt.plot(
                        values,
                        label=f"{pop_size}_{elite_best}_{elite_worst} (worst)",
                        color="red",
                    )
                else:
                    plt.plot(
                        values,
                        color="grey",
                        alpha=0.2,
                    )

        plt.title(f"{value} ({function})")
        plt.xlabel("Iteration")
        plt.ylabel(value)
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    ndim = 2
    os.chdir(f"results_{ndim}D")

    functions = ["F12014"]  # , "F112014"]
    pop_sizes = [50, 100]
    elite_best_values = [0.10, 0.15, 0.20, 0.25, 0.30]
    elite_worst_values = [0.10, 0.15, 0.20, 0.25, 0.30]

    # for function in functions:
    #     for pop_size in pop_sizes:
    #         plot_best(function, pop_size)

    for function in functions:
        plot_line(function, value="list_exploitation")
