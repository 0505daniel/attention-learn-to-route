# YOU SHOULD ACTIVATE THIS LINE WHEN YOUR USING CONDA (NOT RECOMMENDED => TOO SLOW)
from julia.api import Julia
jl = Julia(compiled_modules=False) 

from julia import TSPDrone
from julia import Concorde

def run_tsp_ep(tour, x, y, alpha):
    """
    Python interface for the TSP-ep algorithm.
    """    
    # Generate cost matrix with dummy
    Ct, Cd = TSPDrone.cost_matrices_with_dummy(x, y, 1.0, 1.0 / alpha)

    return TSPDrone.exact_partitioning(tour, Ct, Cd)

def run_concorde(x, y):
    """
    Python interface for the Concorde algorithm.
    """    
    x = x * 10000
    y = y * 10000
    opt_tour, opt_len = Concorde.solve_tsp(x, y, dist="EUC_2D")

    return opt_tour, opt_len / 10000


if __name__ == "__main__":

    import numpy as np

    n = 10
    x = np.random.rand(n)
    y = np.random.rand(n)

    # tour = np.random.permutation(np.arange(2, n + 1))
    # tour = np.insert(tour, 0, 1)
    # tour = np.append(tour, n+1)
    
    opt_tour, opt_len = run_concorde(x, y)
    opt_tour = np.append(opt_tour, n+1)
    print(opt_tour)
    print(opt_len)

    alpha = 2.0
    final_time, truck_route, drone_route = run_tsp_ep(opt_tour, x, y, alpha)

    print(final_time / 10000)
    print(truck_route)
    print(drone_route)