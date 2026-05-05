import os
import networkx as nx
import numpy as np


def wl_hash(G):
    if G.number_of_nodes() == 0:
        return '__empty__'
    return nx.weisfeiler_lehman_graph_hash(G, iterations=3)


def novelty_uniqueness(generated, training):
    train_hashes = {wl_hash(G) for G in training}
    gen_hashes = [wl_hash(G) for G in generated]

    novel = [h not in train_hashes for h in gen_hashes]
    seen = set()
    unique = []
    for h in gen_hashes:
        unique.append(h not in seen)
        seen.add(h)

    novel_and_unique = [n and u for n, u in zip(novel, unique)]
    n = len(generated)
    return {
        'novel': 100.0 * sum(novel) / n,
        'unique': 100.0 * sum(unique) / n,
        'novel_and_unique': 100.0 * sum(novel_and_unique) / n,
    }


def graph_stats(G):
    degrees = [d for _, d in G.degree()]
    clustering = list(nx.clustering(G).values())
    try:
        eig = list(nx.eigenvector_centrality(G, max_iter=1000).values())
    except (nx.PowerIterationFailedConvergence, nx.NetworkXException):
        eig = []
    return degrees, clustering, eig


def evaluate_all(training, baseline_samples, vae_samples):
    os.makedirs('runs', exist_ok=True)

    baseline_metrics = novelty_uniqueness(baseline_samples, training)
    vae_metrics = novelty_uniqueness(vae_samples, training)

    header = f"{'Model':<12} {'Novel%':>8} {'Unique%':>8} {'Nov+Uniq%':>10}"
    sep = '-' * len(header)
    rows = [
        header, sep,
        f"{'Baseline':<12} {baseline_metrics['novel']:>8.1f} {baseline_metrics['unique']:>8.1f} {baseline_metrics['novel_and_unique']:>10.1f}",
        f"{'GraphVAE':<12} {vae_metrics['novel']:>8.1f} {vae_metrics['unique']:>8.1f} {vae_metrics['novel_and_unique']:>10.1f}",
    ]
    table = '\n'.join(rows)
    print(table)
    with open('runs/metrics.txt', 'w') as f:
        f.write(table + '\n')

    return {'baseline': baseline_metrics, 'vae': vae_metrics}


if __name__ == '__main__':
    import pickle
    from data import load_mutag, train_val_test_split, pyg_to_nx

    dataset = load_mutag()
    train_split, _, _ = train_val_test_split(dataset)
    training_graphs = [pyg_to_nx(d) for d in train_split]

    with open('runs/baseline_samples.pkl', 'rb') as f:
        baseline_samples = pickle.load(f)
    with open('runs/vae_samples.pkl', 'rb') as f:
        vae_samples = pickle.load(f)

    evaluate_all(training_graphs, baseline_samples, vae_samples)
