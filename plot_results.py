import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_best():
    # read the data from the csv file
    for f in functions:
        for pop_size in pop_sizes:
            # for one population size i want to plot a 3d graph of bars representing the mean of iterations best fitness value for each elite_best and elite_worst value
            X = np.array(elite_best_values)
            Y = np.array(elite_worst_values)
            X, Y = np.meshgrid(X, Y)
            Z = np.zeros((len(X), len(Y)))

            for elite_best in elite_best_values:
                for elite_worst in elite_worst_values:
                    # read the data from the csv file
                    df = pd.read_csv(f"results_{ndim}D/{f}_{pop_size}_{elite_best}_{elite_worst}.csv")
                    Z[elite_best_values.index(elite_best), elite_worst_values.index(elite_worst)] = df["best_fit"].mean()

            # plot heatmap
            plt.figure()
            # better color map
            plt.imshow(Z, cmap="BuGn", interpolation="nearest")
            plt.colorbar()
            plt.title(f"Mean of best fitness value for {f} with pop_size={pop_size}")
            plt.xlabel("Elite best")
            plt.ylabel("Elite worst")
            plt.xticks(np.arange(len(elite_best_values)), elite_best_values)
            plt.yticks(np.arange(len(elite_worst_values)), elite_worst_values)
            plt.tight_layout()
            plt.show()

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.bar3d(X.ravel(), Y.ravel(), np.zeros_like(Z).ravel(), 0.1, 0.1, Z.ravel(), shade=True)
            ax.set_title(f"Mean of best fitness value for {f} with pop_size={pop_size}")
            ax.set_xlabel("Elite best")
            ax.set_ylabel("Elite worst")
            ax.set_zlabel("Best fitness value")
            plt.tight_layout()
            plt.show()







if __name__ == "__main__":
    # Experiment setting
    ndim = 10
    functions = ["F112014"]  #, "F112014"]
    pop_sizes = [150]  #, 100, 150]
    elite_best_values = [0.1, 0.2, 0.3]
    elite_worst_values = [0.1, 0.2, 0.3]

    plot_best()
