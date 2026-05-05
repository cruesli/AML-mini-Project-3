import numpy as np
import networkx as nx
import torch
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset

N_MAX = 28


def load_mutag(root='data/'):
    return TUDataset(root=root, name='MUTAG')


def train_val_test_split(dataset, sizes=(100, 44, 44), seed=0):
    rng = torch.Generator().manual_seed(seed)
    return random_split(dataset, sizes, generator=rng)


def _degree_sort_order(data):
    n = data.num_nodes
    deg = np.zeros(n, dtype=np.int64)
    if data.edge_index.numel() > 0:
        idx = data.edge_index[1].numpy()
        np.add.at(deg, idx, 1)
    secondary = data.x.argmax(dim=-1).numpy()
    # np.lexsort: last key is primary; ascending -deg = descending degree
    return torch.from_numpy(np.lexsort((secondary, -deg))).long()


def attach_targets(dataset, n_max=N_MAX):
    rows, cols = torch.triu_indices(n_max, n_max, offset=1)
    result = []
    for data in dataset:
        n = data.num_nodes
        order = _degree_sort_order(data)
        inv = torch.empty(n, dtype=torch.long)
        inv[order] = torch.arange(n)

        A = torch.zeros(n_max, n_max)
        if data.edge_index.numel() > 0:
            ei = inv[data.edge_index]
            A[ei[0], ei[1]] = 1.0
            A[ei[1], ei[0]] = 1.0

        new_data = data.clone()
        new_data.A_target = A[rows, cols].float()
        new_data.A_mask = ((rows < n) & (cols < n)).float()
        result.append(new_data)
    return result


def pyg_to_nx(data):
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    if data.edge_index.numel() > 0:
        G.add_edges_from(data.edge_index.t().tolist())
    return G


def empirical_n_distribution(train_subset, n_max=N_MAX):
    counts = np.zeros(n_max + 1)
    for data in train_subset:
        counts[data.num_nodes] += 1
    total = counts.sum()
    return counts / total if total > 0 else counts


if __name__ == '__main__':
    from torch_geometric.loader import DataLoader

    dataset = load_mutag()
    train, val, test = train_val_test_split(dataset)
    print(f'Splits: train={len(train)}, val={len(val)}, test={len(test)}')

    train_data = attach_targets(train)
    val_data = attach_targets(val)

    n_dist = empirical_n_distribution(train)
    print('Empirical N distribution:')
    for n, p in enumerate(n_dist):
        if p > 0:
            print(f'  N={n}: {p:.3f}')

    loader = DataLoader(train_data, batch_size=4)
    batch = next(iter(loader))
    print(f'\nBatch A_target shape: {batch.A_target.shape}')
    print(f'Batch A_mask shape:   {batch.A_mask.shape}')

    print('\nMask sum == N*(N-1)/2 check (first 5 graphs):')
    for i, d in enumerate(train_data[:5]):
        n = d.num_nodes
        expected = n * (n - 1) // 2
        actual = int(d.A_mask.sum().item())
        print(f'  graph {i}: N={n}, mask_sum={actual}, expected={expected}, ok={actual == expected}')
