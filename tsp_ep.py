# YOU SHOULD ACTIVATE THIS LINE WHEN YOUR USING CONDA (NOT RECOMMENDED => TOO SLOW)
from julia.api import Julia
jl = Julia(compiled_modules=False) 
from julia import TSPDrone

def run_tsp_ep(tour, x, y, alpha):
    """
    Python interface for the TSP-ep algorithm.
    """    
    # Generate cost matrix with dummy
    Ct, Cd = TSPDrone.cost_matrices_with_dummy(x, y, 1.0, alpha)
    return TSPDrone.exact_partitioning(tour, Ct, Cd)

if __name__ == "__main__":

    import numpy as np

    n = 10
    x = np.random.rand(n)
    y = np.random.rand(n)
    import pdb; pdb.set_trace()
    tour = np.random.permutation(np.arange(2, n + 1))
    tour = np.insert(tour, 0, 1)
    tour = np.append(tour, n+1)
    print(tour)

    alpha = 2.0
    final_time, truck_route, drone_route = run_tsp_ep(tour, x, y, alpha)

    print(final_time)
    print(truck_route)
    print(drone_route)