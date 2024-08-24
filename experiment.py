import os
import numpy as np
import pandas as pd
from opfunu.cec_based.cec2014 import F12014, F112014
from mealpy import FloatVar, GA


def experiment(function, pop=50, elite_best=0.1, elite_worst=0.3):
    problem_dict = {
        "obj_func": function,
        "bounds": FloatVar(lb=(-100.0,) * ndim, ub=(100.0,) * ndim),
        "minmax": "min",
        "log_to": "None",
        "save_population": True,
    }

    model = GA.EliteSingleGA(
        epoch=100,
        pop_size=pop,
        selection="tournament",
        crossover="uniform",
        elite_best=elite_best,
        elite_worst=elite_worst,
    )

    best = model.solve(problem_dict)

    return (
        best.target.fitness,
        np.array(model.history.list_global_best_fit),
        np.array(model.history.list_diversity),
        np.array(model.history.list_exploration),
        np.array(model.history.list_exploitation),
    )


def run_experiments(
    functions, iterations, pop_sizes, elite_best_values, elite_worst_values
):
    columns = [
        "iteration",
        "function",
        "pop_size",
        "elite_best",
        "elite_worst",
        "best_fit",
        "list_fitness",
        "list_diversity",
        "list_exploitation",
        "list_exploration",
    ]

    for function in functions:
        f = function.__self__.__class__.__name__
        for pop in pop_sizes:
            for elite_best in elite_best_values:
                for elite_worst in elite_worst_values:
                    df = pd.DataFrame(columns=columns).astype(
                        {
                            "iteration": int,
                            "function": str,
                            "pop_size": int,
                            "elite_best": float,
                            "elite_worst": float,
                            "best_fit": float,
                            "list_fitness": object,
                            "list_diversity": object,
                            "list_exploitation": object,
                            "list_exploration": object,
                        }
                    )
                    for iteration in range(iterations):
                        (
                            best,
                            global_best_fit,
                            diversity,
                            exploration,
                            exploitation,
                        ) = experiment(
                            function,
                            pop=pop,
                            elite_best=elite_best,
                            elite_worst=elite_worst,
                        )

                        df = pd.concat(
                            [
                                df,
                                pd.DataFrame(
                                    {
                                        "iteration": iteration,
                                        "function": f,
                                        "pop_size": pop,
                                        "elite_best": elite_best,
                                        "elite_worst": elite_worst,
                                        "best_fit": best,
                                        "list_fitness": [global_best_fit],
                                        "list_diversity": [diversity],
                                        "list_exploitation": [exploitation],
                                        "list_exploration": [exploration],
                                    },
                                    index=[0],
                                ),
                            ]
                        )
                    df.to_csv(
                        f"{f}_{pop}_{elite_best}_{elite_worst}.csv",
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
    pop_sizes = [50, 100, 150]
    elite_best_values = [0.1, 0.2, 0.3]
    elite_worst_values = [0.1, 0.2, 0.3]

    run_experiments(
        functions, iterations, pop_sizes, elite_best_values, elite_worst_values
    )
