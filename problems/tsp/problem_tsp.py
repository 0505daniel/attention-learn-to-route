from torch.utils.data import Dataset
import torch
import os
import pickle
from problems.tsp.state_tsp import StateTSP
from utils.beam_search import beam_search
from tsp_ep import run_tsp_ep_batch
import numpy as np

def shift_left_to_right(arr):
    zero_idx = np.where(arr == 0)[0][0]
    arr = np.concatenate((arr[zero_idx:], arr[:zero_idx]))
    num_nodes = len(arr)
    arr = np.append(arr, num_nodes)
    return arr

def preprocessing(tour):
    batch_size = tour.shape[0]
    tour_list = []
    for i in range(batch_size):
        tour_list.append(shift_left_to_right(tour[i]))
    tour_np = np.array(tour_list)
    return tour_np


class TSP(object):

    NAME = 'tsp'

    @staticmethod
    def get_costs(dataset, pi):
        # Check that tours are valid, i.e. contain 0 to n -1
        assert (
            torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) ==
            pi.data.sort(1)[0]
        ).all(), "Invalid tour"

        # Gather dataset in order of tour
        d = dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset))

        #WARNING: Julia is 1-indexed, so if you want to run the function, I think you should +1 the tour
        #WARNING: For the function run_tsp_ep, the tour should be 1 to n + 1, where 1 is the first and n+1 is the last
        
        #NOTE: Get TSP-D cost
        x = dataset[:, :, 0].cpu().numpy()
        y = dataset[:, :, 1].cpu().numpy()
        tour = pi[:].cpu().numpy()
        tour = preprocessing(tour)
        alpha = 2.0
        costs, truck_route, drone_route = run_tsp_ep_batch(tour, x, y, alpha)
        import gc; gc.collect()
        return torch.FloatTensor(costs).to(pi.device), None
        
        # Length is distance (L2-norm of difference) from each next location from its prev and of last from first
        #return (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - d[:, -1]).norm(p=2, dim=1), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return TSPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateTSP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = TSP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


class TSPDataset(Dataset):
    
    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None):
        super(TSPDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [torch.FloatTensor(row) for row in (data[offset:offset+num_samples])]
        else:
            # Sample points randomly in [0, 1] square
            self.data = [torch.FloatTensor(size, 2).uniform_(0, 1) for i in range(num_samples)]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
