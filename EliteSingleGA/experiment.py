import os
import numpy as np
import pandas as pd
from opfunu.cec_based.cec2014 import F12014, F112014
from mealpy import FloatVar, GA


def experiment(
    function, epoch=1000, pop_size=50, elite_best=0.1, elite_worst=0.3
):
    problem_dict = {
        "obj_func": function,
        "bounds": FloatVar(lb=(-100.0,) * ndim, ub=(100.0,) * ndim),
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


def run_experiments(
    functions, iterations, pop_sizes, elite_best_values, elite_worst_values
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
        f = function.__self__.__class__.__name__
        for pop_size in pop_sizes:
            for elite_best in elite_best_values:
                for elite_worst in elite_worst_values:
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
                        ) = experiment(
                            function,
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
                                        "list_exploitation": [
                                            list_exploitation
                                        ],
                                        "list_diversity": [list_diversity],
                                    },
                                    index=[0],
                                ),
                            ]
                        )
                    df.to_csv(
                        f"{f}_{pop_size}_{elite_best}_{elite_worst}.csv",
                        index=False,
                    )


if __name__ == "__main__":
    ndim = 10
    f1 = F12014(ndim=ndim).evaluate
    f11 = F112014(ndim=ndim).evaluate

    # if plots results does not exist, create it
    if not os.path.exists(f"results_{ndim}D"):
        os.makedirs(f"results_{ndim}D")
    os.chdir(f"results_{ndim}D")

    # Experiment setting
    iterations = 10
    functions = [f1, f11]
    pop_sizes = [50, 100, 150, 200]
    elite_best_values = [0.10, 0.15, 0.20, 0.25, 0.30]
    elite_worst_values = [0.10, 0.15, 0.20, 0.25, 0.30]

    run_experiments(
        functions, iterations, pop_sizes, elite_best_values, elite_worst_values
    )
