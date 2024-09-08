import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_best(function, pop_size):
    df = pd.read_csv(
        # f"{function}_{pop_size}_{elite_best}_{elite_worst}_GA.csv"
        f"{function}_{pop_size}_{elite_best}_{elite_worst}_ABC.csv"
    )
    best_fitness_values = df["best_fitness"].values
    
    # Plotando os valores de best_fitness como um gráfico de linha
    plt.figure()
    plt.plot(best_fitness_values, label="Best Fitness", color="blue")
    plt.title(f"Best Fitness Over Iterations ({function}, pop_size={pop_size})")
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_line(function, value="list_fitness"):
    df = pd.read_csv(
        # f"{function}_{pop_size}_{elite_best}_{elite_worst}_GA.csv"
        f"{function}_{pop_size}_{elite_best}_{elite_worst}_ABC.csv"
    )
    values = np.array([np.fromstring(i[1:-1], sep=" ") for i in df[value].values])
    
    # Calcular a média dos valores ao longo do eixo 0
    mean_values = values.mean(axis=0)

    # Encontrar os índices dos melhores e piores elites
    best_elite_index = mean_values.argmin()  # Índice do menor valor
    worst_elite_index = mean_values.argmax()  # Índice do maior valor

    # Plotar os valores médios
    plt.figure()
    
    # Determinar as cores e os rótulos para o melhor e pior
    if elite_best == best_elite_index and elite_worst == worst_elite_index:
        plt.plot(mean_values, label=f"{pop_size}_{elite_best}_{elite_worst} (best)", color="green")
    elif elite_best == worst_elite_index and elite_worst == best_elite_index:
        plt.plot(mean_values, label=f"{pop_size}_{elite_best}_{elite_worst} (worst)", color="red")
    else:
        plt.plot(mean_values, color="blue", alpha=0.5)

    # Título e rótulos
    plt.title(f"{value} ({function})")
    plt.xlabel("Iteration")
    plt.ylabel(value)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ndim = 2
    os.chdir(f"results_{ndim}D")

    functions = ["F52014", "F92014"]
    pop_size = 50
    elite_best = 0.2
    elite_worst = 0.2

    # for function in functions:
    #     plot_best(function, pop_size)

    for function in functions:
        plot_line(function, value="list_exploitation")
