from mealpy import ABC, GA, FloatVar
from opfunu.cec_based.cec2014 import F52014, F92014

def experiment(obj_func, ndim):

    problem_dict = {
        "bounds": FloatVar(lb=(-100.0,) * ndim, ub=(100.0,) * ndim),
        "minmax": "min",
        "obj_func": obj_func,
        "log_to": "None",
        "save_population": True,
    }

    modelABC = ABC.OriginalABC(epoch=1000)
    modelGA = GA.EliteSingleGA(epoch=1000)
    bestABC = modelABC.solve(problem_dict)
    bestGA = modelGA.solve(problem_dict)
    return bestABC, bestGA, modelABC, modelGA

if __name__ == "__main__":

    ndim = 10
    funcs = [("F5", F52014(ndim).evaluate), ("F9", F92014(ndim).evaluate)]

    iterations = 10

    for (func_name, obj_func) in funcs:
        for i in range(iterations):
            bestABC, bestGA, modelABC, modelGA = experiment(obj_func, ndim)
            print("Function: %s, Iteration: %d, ABC: %.2f, GA: %.2f" % (func_name, i, bestABC.target.fitness, bestGA.target.fitness))
