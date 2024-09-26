import os
import numpy as np
import pandas as pd
from mealpy import FloatVar, GA, ABC
from problem import evaluate, bounds


def experiment_GA(
    function, bounds, epoch=1000, pop_size=50, elite_best=0.2, elite_worst=0.2
):
    problem_dict = {
        "obj_func": function,
        "bounds": bounds,
        "minmax": "min",
        "log_to": "None",
        "save_population": True,
    }

    model = GA.EliteSingleGA(
        epoch=epoch,
        pop_size=pop_size,
        selection="tournament",
        crossover="uniform",
        elite_best=elite_best,
        elite_worst=elite_worst,
    )

    best = model.solve(problem_dict)

    return (
        best.target.fitness,
        best.solution,
        np.array(model.history.list_global_best_fit),
        np.array(model.history.list_exploration),
        np.array(model.history.list_exploitation),
        np.array(model.history.list_diversity),
    )


def experiment_ABC(
    function, bounds, epoch=1000, pop_size=50, elite_best=0.2, elite_worst=0.2
):
    problem_dict = {
        "obj_func": function,
        "bounds": bounds,
        "minmax": "min",
        "log_to": "None",
        "save_population": True,
    }

    model = ABC.OriginalABC(
        epoch=epoch,
        pop_size=pop_size,
        selection="tournament",
        crossover="uniform",
        elite_best=elite_best,
        elite_worst=elite_worst,
    )

    best = model.solve(problem_dict)

    return (
        best.target.fitness,
        best.solution,
        np.array(model.history.list_global_best_fit),
        np.array(model.history.list_exploration),
        np.array(model.history.list_exploitation),
        np.array(model.history.list_diversity),
    )


def run_experiments_GA(
    functions, bounds, iterations, pop_size, elite_best, elite_worst
):
    columns = [
        "best_fitness",
        "best_solution",
        "list_fitness",
        "list_exploration",
        "list_exploitation",
        "list_diversity",
    ]

    for function in functions:
        f = "ballbearingshit"
        df = pd.DataFrame(columns=columns).astype(
            {
                "best_fitness": float,
                "best_solution": object,
                "list_fitness": object,
                "list_exploration": object,
                "list_exploitation": object,
                "list_diversity": object,
            }
        )
        for iteration in range(iterations):
            (
                best_fitness,
                best_solution,
                list_fitness,
                list_exploitation,
                list_exploration,
                list_diversity,
            ) = experiment_GA(
                function,
                bounds,
                pop_size=pop_size,
                elite_best=elite_best,
                elite_worst=elite_worst,
            )

            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        {
                            "best_fitness": best_fitness,
                            "best_solution": [best_solution],
                            "list_fitness": [list_fitness],
                            "list_exploration": [list_exploration],
                            "list_exploitation": [list_exploitation],
                            "list_diversity": [list_diversity],
                        },
                        index=[0],
                    ),
                ]
            )
        df.to_csv(
            f"{f}_{pop_size}_{elite_best}_{elite_worst}_GA.csv",
            index=False,
        )


def run_experiments_ABC(
    functions, bounds, iterations, pop_size, elite_best, elite_worst
):
    columns = [
        "best_fitness",
        "best_solution",
        "list_fitness",
        "list_exploration",
        "list_exploitation",
        "list_diversity",
    ]

    for function in functions:
        f = "ballbearingshit"
        df = pd.DataFrame(columns=columns).astype(
            {
                "best_fitness": float,
                "best_solution": object,
                "list_fitness": object,
                "list_exploration": object,
                "list_exploitation": object,
                "list_diversity": object,
            }
        )
        for iteration in range(iterations):
            (
                best_fitness,
                best_solution,
                list_fitness,
                list_exploitation,
                list_exploration,
                list_diversity,
            ) = experiment_ABC(
                function,
                bounds,
                pop_size=pop_size,
                elite_best=elite_best,
                elite_worst=elite_worst,
            )

            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        {
                            "best_fitness": best_fitness,
                            "best_solution": [best_solution],
                            "list_fitness": [list_fitness],
                            "list_exploration": [list_exploration],
                            "list_exploitation": [list_exploitation],
                            "list_diversity": [list_diversity],
                        },
                        index=[0],
                    ),
                ]
            )
        df.to_csv(
            f"{f}_{pop_size}_{elite_best}_{elite_worst}_ABC.csv",
            index=False,
        )


if __name__ == "__main__":
    f = evaluate

    # if plots results does not exist, create it
    if not os.path.exists("results"):
        os.makedirs("results")
    os.chdir("results")

    # Experiment setting (para outras configs checar tag trab2)
    functions = [f]
    iterations = 10
    pop_size = 200
    elite_best = 0.2
    elite_worst = 0.2

    run_experiments_GA(
        functions, bounds, iterations, pop_size, elite_best, elite_worst
    )
    run_experiments_ABC(
        functions, bounds, iterations, pop_size, elite_best, elite_worst
    )
