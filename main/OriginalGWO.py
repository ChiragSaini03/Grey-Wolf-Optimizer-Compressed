import os
import opfunu
import numpy as np
import pandas as pd
from mealpy.optimizer import Optimizer
from mealpy import FloatVar
from mealpy.swarm_based.GWO import OriginalGWO
import sys

benchmark_dict = {
    2017: range(1, 30),
    2021: range(1, 10),
    2022: range(1, 13),
}

directory_path = f"./results/OriginalGWO"
if not os.path.exists(directory_path):
    os.makedirs(directory_path)

def final_result(model_name):
    WOAResults = {}
    for year in benchmark_dict.keys():
        for func_num in benchmark_dict[year]:
            func_map_key_name = f"F{func_num}{year}"
            WOAResults[func_map_key_name] = {}
            for dimension in [10, 30, 50, 100]:
                func_name = f"opfunu.cec_based.F{func_num}{year}(ndim={dimension})"
                WOAResults[func_map_key_name][dimension] = {}
                problem = eval(func_name)
                problem_dict = {
                    "bounds": FloatVar(lb=problem.lb, ub=problem.ub, name="delta"),
                    "minmax": "min",
                    "obj_func": problem.evaluate,
                    "n_dims": dimension,
                    "log_to": "file",
                    "log_file": "history-original.log",
                }
                for pop_size in [
                    10,
                    20,
                    30,
                    40,
                    50,
                    70,
                    100,
                    200,
                    300,
                    400,
                    500,
                    700,
                    1000,
                ]:
                    run_result = np.array([])
                    for iter in range(1, 6):
                        epoch = 10000
                        pop_size = pop_size
                        model = model_name(epoch, pop_size)
                        max_fe = 10000 * dimension
                        term_dict = {"max_fe": max_fe}
                        best_position = model.solve(
                            problem_dict,
                            n_workers=3
                        )
                        best_fitness = model.history.list_global_best_fit[-1]
                        # print(best_fitness)
                        run_result = np.append(run_result, best_fitness)
                        print(f"{model.name} => Function: {func_map_key_name}, Dimension: {dimension}, Population Size: {pop_size} :: Iteration: {iter}, Fitness: {best_fitness}",flush=True,)
                    data = np.mean(run_result)
                    WOAResults[func_map_key_name][dimension][pop_size] = data
                    print(f"=========={model.name} => Function: {func_map_key_name}, Dimension: {dimension}, Population Size: {pop_size} :: fitness: {data} ::  Completed==========",
                        flush=True,
                    )
                    if not os.path.exists(f"{directory_path}/{func_map_key_name}"):
                        os.makedirs(f"{directory_path}/{func_map_key_name}")
                        if not os.path.exists(f"{directory_path}/{func_map_key_name}/states"):
                            os.makedirs(f"{directory_path}/{func_map_key_name}/states")
                    with open(f"{directory_path}/{func_map_key_name}/states/d{dimension}ps{pop_size}.txt", "w") as f:
                        f.write(str(data))
            df = pd.DataFrame(WOAResults[func_map_key_name])
            df.to_csv(f"{directory_path}/{func_map_key_name}.csv", index=True)


with open("./progress-original.txt", "w") as f:
    final_result(OriginalGWO)
