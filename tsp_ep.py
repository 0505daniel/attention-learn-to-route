# YOU SHOULD ACTIVATE THIS LINE WHEN YOUR USING CONDA (NOT RECOMMENDED => TOO SLOW)
from julia.api import Julia
jl = Julia(compiled_modules=False) 

from julia import TSPDrone

def run_tsp_ep(x, y, alpha):
    """
    Python interface for the TSP-ep algorithm.
    """
    return TSPDrone.solve_tspd(x, y, 1.0, 1.0 / alpha, n_groups=1, method="TSP-ep")

if __name__ == "__main__":

    import numpy as np

    n = 100
    x = np.random.rand(n)
    y = np.random.rand(n)

    alpha = 2.0
    result = run_tsp_ep(x, y, alpha)
    
    print(result.total_cost)
    print(result.truck_route)
    print(result.drone_route)