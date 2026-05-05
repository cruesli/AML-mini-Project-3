import pickle
import os
import torch
import numpy as np

from data import load_mutag, train_val_test_split, empirical_n_distribution
from baseline import ErdosRenyiBaseline
from model import build_model


def sample_baseline(train_dataset, num=1000, seed=0):
    os.makedirs('runs', exist_ok=True)
    baseline = ErdosRenyiBaseline()
    baseline.fit(train_dataset)
    rng = np.random.default_rng(seed)
    graphs = baseline.sample(num, rng)
    with open('runs/baseline_samples.pkl', 'wb') as f:
        pickle.dump(graphs, f)
    return graphs


def sample_vae(model_path, n_dist, num=1000, seed=0, device='cpu'):
    os.makedirs('runs', exist_ok=True)
    torch.manual_seed(seed)
    model = build_model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    graphs = model.sample(num, n_dist, device=device)
    with open('runs/vae_samples.pkl', 'wb') as f:
        pickle.dump(graphs, f)
    return graphs


if __name__ == '__main__':
    dataset = load_mutag()
    train_split, _, _ = train_val_test_split(dataset)
    n_dist = empirical_n_distribution(train_split)

    print('Sampling baseline...')
    baseline_graphs = sample_baseline(train_split, num=1000, seed=0)
    print(f'  {len(baseline_graphs)} graphs, mean N={sum(g.number_of_nodes() for g in baseline_graphs)/len(baseline_graphs):.2f}')

    print('Sampling VAE...')
    vae_graphs = sample_vae('runs/model.pt', n_dist, num=1000, seed=0)
    print(f'  {len(vae_graphs)} graphs, mean N={sum(g.number_of_nodes() for g in vae_graphs)/len(vae_graphs):.2f}')
