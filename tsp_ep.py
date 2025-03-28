import numpy as np
from joblib import Parallel, delayed

def travel_cost(path, C):
    """
    Compute the travel cost along a path given the cost matrix C.
    
    Parameters:
      path : list of node indices (assumed 0-indexed)
      C    : 2D NumPy array representing the cost matrix
      
    Returns:
      Total cost as a float.
    """
    total = 0.0
    for i in range(len(path) - 1):
        total += C[path[i], path[i+1]]
    return total


def objective_value(truck_route, drone_route, Ct, Cd):
    """
    Compute the objective value from truck and drone routes.
    The objective value is the sum over segments of the maximum 
    of the truck and drone travel costs.
    """
    # Determine the combined nodes in the order they appear in truck_route.
    combined_nodes = [node for node in truck_route if node in drone_route]
    obj_val = 0.0
    for i in range(len(combined_nodes) - 1):
        j1 = combined_nodes[i]
        j2 = combined_nodes[i+1]
        try:
            t_idx1 = truck_route.index(j1)
            t_idx2 = truck_route.index(j2)
        except ValueError:
            continue
        t_cost = travel_cost(truck_route[t_idx1:t_idx2+1], Ct)
        d_idx1 = drone_route.index(j1)
        d_idx2 = drone_route.index(j2)
        d_cost = travel_cost(drone_route[d_idx1:d_idx2+1], Cd)
        obj_val += max(t_cost, d_cost)
    return obj_val


def cost_matrices_with_dummy_from_matrices(truck_cost_mtx, drone_cost_mtx):
    """
    Construct cost matrices with a dummy node from given truck and drone cost matrices.

    This function mimics the Julia version:
    
        Ct = [ truck_cost_mtx          truck_cost_mtx[:, 1];
               truck_cost_mtx[1, :]'    0.0 ]
    
        Cd = [ drone_cost_mtx          drone_cost_mtx[:, 1];
               drone_cost_mtx[1, :]'    0.0 ]
    
    Parameters:
      truck_cost_mtx : 2D NumPy array for truck costs
      drone_cost_mtx : 2D NumPy array for drone costs
      
    Returns:
      Tuple (Ct, Cd) where each is a new NumPy array with the dummy node appended.
    """
    Ct = np.block([
        [truck_cost_mtx, truck_cost_mtx[:, [0]]],
        [truck_cost_mtx[0, :][np.newaxis, :], np.array([[0.0]])]
    ])
    
    Cd = np.block([
        [drone_cost_mtx, drone_cost_mtx[:, [0]]],
        [drone_cost_mtx[0, :][np.newaxis, :], np.array([[0.0]])]
    ])
    
    return Ct, Cd


def _cost_matrices_with_dummy(x, y, speed_of_truck, speed_of_drone):
    """
    Compute the cost matrices Ct and Cd from coordinate arrays x and y and speeds.
    
    This function calculates the Euclidean distance between each pair of nodes,
    then multiplies by the given speeds.
    
    Parameters:
      x, y          : lists or arrays of coordinates (must be the same length)
      speed_of_truck: scalar multiplier for the truck cost matrix
      speed_of_drone: scalar multiplier for the drone cost matrix
      
    Returns:
      Tuple (Ct, Cd) as 2D NumPy arrays.
    """
    x = np.array(x)
    y = np.array(y)
    n_nodes = len(x)
    assert len(x) == len(y), "x and y must have the same length"
    
    dist = np.zeros((n_nodes, n_nodes), dtype=float)
    for i in range(n_nodes):
        for j in range(n_nodes):
            dist[i, j] = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
    
    Ct = speed_of_truck * dist
    Cd = speed_of_drone * dist
    return Ct, Cd


def cost_matrices_with_dummy(x, y, speed_of_truck, speed_of_drone):
    """
    Append the starting node to the coordinate lists and compute the cost matrices.
    
    Parameters:
      x, y          : lists or arrays of coordinates for the nodes
      speed_of_truck: scalar multiplier for the truck cost matrix
      speed_of_drone: scalar multiplier for the drone cost matrix
      
    Returns:
      Tuple (Ct, Cd) as computed by _cost_matrices_with_dummy.
    """
    xx = list(x)
    yy = list(y)
    xx.append(x[0])
    yy.append(y[0])
    return _cost_matrices_with_dummy(xx, yy, speed_of_truck, speed_of_drone)


def check_operation_(r, T, M, i, j, k, Cd, SUM, flying_range=1e6):
    """
    Evaluate the potential improvement for moving from i to j via k.
    r    : list or array of node indices (0-indexed)
    T    : 2D NumPy array of current best times/costs
    M    : 2D NumPy array for recording intermediate decisions
    Cd   : 2D NumPy array (drone cost matrix)
    SUM  : 3D NumPy array used for intermediate truck route cost sums
    i, j, k : integer indices (assumed already 0-indexed)
    flying_range : maximum allowed drone range
    """
    Tk1 = Cd[r[i], r[k]] + Cd[r[k], r[j]]
    if Tk1 <= flying_range:
        Tk2 = SUM[i, j, k]
        Tk = max(Tk1, Tk2)
        if Tk < T[i, j]:
            T[i, j] = Tk
            M[r[i], r[j]] = r[k]


def exact_partitioning(initial_tour, Ct, Cd, flying_range=1e6, complexity=3):
    """
    Performs exact partitioning of a given tour using either a naive O(n^3)
    method (complexity == 4) or an alternative recursive version.
    
    Parameters:
      initial_tour : list of node indices (0-indexed)
      Ct           : 2D NumPy array for truck cost matrix
      Cd           : 2D NumPy array for drone cost matrix
      flying_range : maximum drone range allowed
      complexity   : set to 4 for naive computation; otherwise uses recursion
      
    Returns:
      final_time   : computed objective value (float)
      truck_route  : list of node indices for the truck route
      drone_route  : list of node indices for the drone route
    """
    n = Ct.shape[0]
    r = initial_tour[:]  # copy of initial tour
    T = np.full((n, n), np.inf)
    M = np.full((n, n), -99, dtype=int)

    if complexity == 4:
        # Naive O(n^3) computation
        for i in range(0, n-1):
            for j in range(i+1, n):
                if j == i + 1:
                    T[i, j] = Ct[r[i], r[j]]
                    M[r[i], r[j]] = -1
                else:
                    for k in range(i+1, j):
                        Tk1 = Cd[r[i], r[k]] + Cd[r[k], r[j]]
                        if Tk1 <= flying_range:
                            Tk2 = 0.0
                            # Sum costs from r[i] to r[k-1]
                            for l in range(i, k-1):
                                Tk2 += Ct[r[l], r[l+1]]
                            # Add bypass cost at k
                            Tk2 += Ct[r[k-1], r[k+1]]
                            # Sum costs from r[k+1] to r[j]
                            for l in range(k+1, j):
                                Tk2 += Ct[r[l], r[l+1]]
                            Tk = max(Tk1, Tk2)
                            if Tk < T[i, j]:
                                T[i, j] = Tk
                                M[r[i], r[j]] = r[k]
    else:
        # Recursive version (Appendix method)
        SUM = np.zeros((n, n, n), dtype=float)  # truck route bypassing k
        for i in range(0, n-1):
            for j in range(i+1, n):
                if j == i + 1:
                    T[i, j] = Ct[r[i], r[j]]
                    M[r[i], r[j]] = -1
        for k in range(1, n-1):  # corresponds to Julia's 2:n-1
            for i in range(k-1, -1, -1):
                if i == k-1:
                    SUM[i, k+1, k] = Ct[r[i], r[k+1]]
                else:
                    SUM[i, k+1, k] = Ct[r[i], r[i+1]] + SUM[i+1, k+1, k]
                check_operation_(r, T, M, i, k+1, k, Cd, SUM, flying_range=flying_range)
                for j in range(k+1, n):
                    if SUM[i, j, k] == 0:
                        SUM[i, j, k] = SUM[i, j-1, k] + Ct[r[j-1], r[j]]
                        check_operation_(r, T, M, i, j, k, Cd, SUM, flying_range=flying_range)

    # Dynamic programming to compute optimal objective
    V = np.zeros(n, dtype=float)
    P = np.full(n, -1, dtype=int)

    V[0] = 0
    for i in range(1, n):
        VV = [V[k] + T[k, i] for k in range(i)]
        V[i] = min(VV)
        P[i] = r[np.argmin(VV)]

    # Retrieve the combined solution (truck+drone nodes)
    combined_nodes = []
    current_idx = n - 1
    current = r[current_idx]
    while current != -1:
        combined_nodes.insert(0, current)
        current_idx = list(r).index(current)
        current = P[current_idx]

    # Separate the drone-only nodes from the combined route
    drone_only_nodes = []
    drone_route = [combined_nodes[0]]
    assert combined_nodes[0] == r[0]
    for i in range(len(combined_nodes) - 1):
        j1 = combined_nodes[i]
        j2 = combined_nodes[i+1]
        if M[j1, j2] != -1:
            drone_only_nodes.append(M[j1, j2])
            drone_route.append(M[j1, j2])
        drone_route.append(j2)
    truck_route = [node for node in r if node not in drone_only_nodes]

    obj_val = objective_value(truck_route, drone_route, Ct, Cd)
    final_time = V[-1]

    if not np.isclose(obj_val, final_time, atol=1e-4):
        raise AssertionError("Objective value does not match final time.")

    return final_time, truck_route, drone_route


def run_tsp_ep(tour, x, y, alpha):
    """
    Interface for Python version of TSP-ep algorithm.
    
    Parameters:
      tour  : 1D array-like of size (dim + 1)
      x, y  : 1D array-like of node coordinates, size (dim,)
      alpha : scalar, drone speed factor (if alpha=2.0 -> drone speed = 1/alpha=0.5)
    
    Returns:
      final_time, truck_route, drone_route
    """
    # Generate cost matrix with dummy
    Ct, Cd = cost_matrices_with_dummy(x, y, 1.0, 1.0 / alpha)

    return exact_partitioning(tour, Ct, Cd)


def run_tsp_ep_batch(tours, xs, ys, alphas, n_jobs=-1):
    """
    배치(batch) 형태의 (tour, x, y, alpha) 데이터를
    병렬로 run_tsp_ep 함수를 호출하여 결과를 반환하는 함수.

    Parameters:
      tours   : shape (batch_size, dim+1)
      xs      : shape (batch_size, dim)
      ys      : shape (batch_size, dim)
      alphas  : shape (batch_size,) (혹은 broadcast 가능)
      n_jobs  : joblib Parallel에 넘길 프로세스 혹은 스레드 개수 (기본 -1: 모든 코어 사용)

    Returns:
      final_times   : 길이 batch_size 의 1D np.array
      truck_routes  : 길이 batch_size 의 list of list
      drone_routes  : 길이 batch_size 의 list of list
    """
    alphas = np.ones(xs.shape[0])*alphas
    # 병렬화를 위해 joblib 사용
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_tsp_ep)(tour, x, y, alpha) 
        for tour, x, y, alpha in zip(tours, xs, ys, alphas)
    )
    
    # results는 [(final_time, truck_route, drone_route), ( ... ), ... ] 형태
    final_times, truck_routes, drone_routes = zip(*results)
    
    # final_times는 float이므로 np.array로 묶고,
    # truck_routes와 drone_routes는 길이가 다를 수 있으므로 list of list 형태 그대로 반환
    return np.array(final_times), list(truck_routes), list(drone_routes)


if __name__ == "__main__":

    # === 1) 단일 실행 예시 ===
    x = np.array([
        0.41647154, 0.9528754, 0.2846341, 0.49612117, 0.948007,
        0.19722569, 0.5520318, 0.16971296, 0.4616382,  0.10158819,
        0.6514677,  0.67163014, 0.30766296, 0.46783316, 0.7966255,
        0.65731454, 0.10641873, 0.50887656, 0.14265233, 0.81040597
    ])
    y = np.array([
        0.4943713,  0.11617166, 0.36576557, 0.19983506, 0.03876638,
        0.55138904, 0.37202078, 0.2803604,  0.03344417, 0.40782398,
        0.88867146, 0.74739623, 0.83490664, 0.21177989, 0.14606476,
        0.99657464, 0.1006282,  0.76188064, 0.8084778,  0.2958197
    ])
    tour = np.array([
        1, 13, 4, 14, 9, 19, 3, 6, 8, 10, 17, 2, 5, 16, 11, 12, 20, 15, 18, 7, 21
    ])
    tour -= 1  # 0-index

    alpha = 2.0

    final_time, truck_route, drone_route = run_tsp_ep(tour, x, y, alpha)
    print("=== 단일 실행 결과 ===")
    print("final_time:", final_time)
    print("truck_route:", truck_route)
    print("drone_route:", drone_route)

    # === 2) 배치(batch) 실행 예시 (batch_size=1) ===
    # 모양을 맞추기 위해 (1, 20), (1, 20), (1, 21) 형태로 reshape
    xs = x[np.newaxis, :]
    ys = y[np.newaxis, :]
    tours = tour[np.newaxis, :]
    alphas = np.array([alpha])  # shape: (1,)

    final_times_batch, truck_routes_batch, drone_routes_batch = run_tsp_ep_batch(tours, xs, ys, alphas, n_jobs=-1)
    print("\n=== 배치(batch) 실행 결과 (batch_size=1) ===")
    print("final_time:", final_times_batch[0])
    print("truck_route:", truck_routes_batch[0])
    print("drone_route:", drone_routes_batch[0])
