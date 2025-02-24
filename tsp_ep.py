# YOU SHOULD ACTIVATE THIS LINE WHEN YOUR USING CONDA (NOT RECOMMENDED => TOO SLOW)
from julia.api import Julia
jl = Julia(compiled_modules=False) 
from julia import TSPDrone
from julia import Concorde
import numpy as np
def run_tsp_ep(tour, x, y, alpha):
    x, y = x.astype(np.float64), y.astype(np.float64)
    """
    Python interface for the TSP-ep algorithm.
    """    
    # Generate cost matrix with dummy
    Ct, Cd = TSPDrone.cost_matrices_with_dummy(x, y, 1.0, 1.0 / alpha)

    return TSPDrone.exact_partitioning(tour, Ct, Cd)

# def run_tsp_ep(tour, x, y, alpha):
#     """
#     Python interface for the TSP-ep algorithm.
#     """
#     # Generate cost matrix with dummy
#     Ct, Cd = TSPDrone.cost_matrices_with_dummy(x, y, 1.0, alpha)
#     return TSPDrone.exact_partitioning(tour, Ct * 10000, Cd * 10000)

def run_concorde(x, y):
    x, y = x.astype(np.float64), y.astype(np.float64)
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
    import pdb; pdb.set_trace()
    x = np.random.rand(n)
    y = np.random.rand(n)
    # tour = np.random.permutation(np.arange(2, n + 1))
    # tour = np.insert(tour, 0, 1)
    # tour = np.append(tour, n+1)
    opt_tour, opt_len = run_concorde(x, y)
    opt_tour = np.append(opt_tour, n+1)
    print(opt_tour)
    print(opt_len)
    # x = np.array([0.41647154, 0.9528754, 0.2846341, 0.49612117, 0.948007, 0.19722569,
    #           0.5520318, 0.16971296, 0.4616382, 0.10158819, 0.6514677, 0.67163014,
    #           0.30766296, 0.46783316, 0.7966255, 0.65731454, 0.10641873, 0.50887656,
    #           0.14265233, 0.81040597])
    # y = np.array([0.4943713, 0.11617166, 0.36576557, 0.19983506, 0.03876638, 0.55138904,
    #             0.37202078, 0.2803604, 0.03344417, 0.40782398, 0.88867146, 0.74739623,
    #             0.83490664, 0.21177989, 0.14606476, 0.99657464, 0.1006282, 0.76188064,
    #             0.8084778, 0.2958197])
    # tour = np.array([1, 13, 4, 14, 9, 19, 3, 6, 8, 10, 17, 2, 5, 16, 11, 12, 20, 15, 18, 7, 21])
    alpha = 2.0
    final_time, truck_route, drone_route = run_tsp_ep(opt_tour, x, y, alpha)
    print(final_time / 10000)
    print(truck_route)
    print(drone_route)