import numpy as np
import networkx as nx
from collections import defaultdict

from data import load_mutag, train_val_test_split, pyg_to_nx, N_MAX


class ErdosRenyiBaseline:
    def fit(self, train_dataset):
        edge_counts = defaultdict(list)
        for data in train_dataset:
            n = data.num_nodes
            G = pyg_to_nx(data)
            possible = n * (n - 1) / 2
            density = G.number_of_edges() / possible if possible > 0 else 0.0
            edge_counts[n].append(density)

        # P(N) and per-N mean density
        n_counts = np.zeros(N_MAX + 1)
        self.r_n = {}
        for n, densities in edge_counts.items():
            n_counts[n] = len(densities)
            self.r_n[n] = float(np.mean(densities))

        total = n_counts.sum()
        self.n_probs = n_counts / total if total > 0 else n_counts

    def sample(self, num_graphs, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        ns = np.arange(N_MAX + 1)
        graphs = []
        for _ in range(num_graphs):
            n = int(rng.choice(ns, p=self.n_probs))
            r = self.r_n.get(n, 0.0)
            G = nx.Graph()
            G.add_nodes_from(range(n))
            for u in range(n):
                for v in range(u + 1, n):
                    if rng.random() < r:
                        G.add_edge(u, v)
            graphs.append(G)
        return graphs


if __name__ == '__main__':
    dataset = load_mutag()
    train, _, _ = train_val_test_split(dataset)

    baseline = ErdosRenyiBaseline()
    baseline.fit(train)

    train_ns = [data.num_nodes for data in train]
    train_edges = [pyg_to_nx(data).number_of_edges() for data in train]
    print(f'Training  — mean N: {np.mean(train_ns):.2f}, mean edges: {np.mean(train_edges):.2f}')

    rng = np.random.default_rng(0)
    samples = baseline.sample(10, rng)
    sample_ns = [G.number_of_nodes() for G in samples]
    sample_edges = [G.number_of_edges() for G in samples]
    print(f'Sampled 10 — mean N: {np.mean(sample_ns):.2f}, mean edges: {np.mean(sample_edges):.2f}')
