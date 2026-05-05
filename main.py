import argparse
import random
import numpy as np
import torch

from data import load_mutag, train_val_test_split, attach_targets, empirical_n_distribution, pyg_to_nx
from baseline import ErdosRenyiBaseline
from model import build_model
from train import train as train_model, plot_loss_curve
from sample import sample_baseline, sample_vae
from evaluate import evaluate_all
from plots import plot_stat_grid


def cmd_train(args):
    from torch_geometric.loader import DataLoader

    dataset = load_mutag()
    train_split, val_split, _ = train_val_test_split(dataset)
    train_data = attach_targets(train_split)
    val_data = attach_targets(val_split)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=len(val_data))

    model = build_model(
        latent_dim=args.latent_dim,
        state_dim=args.state_dim,
        num_message_passing_rounds=args.num_message_passing_rounds,
    ).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)

    history = train_model(
        model, optimizer, scheduler, train_loader, val_loader,
        epochs=args.epochs,
        beta_max=args.beta_max,
        warmup_epochs=args.warmup_epochs,
        device=args.device,
    )
    plot_loss_curve(history)


def cmd_sample(args):
    dataset = load_mutag()
    train_split, _, _ = train_val_test_split(dataset)
    n_dist = empirical_n_distribution(train_split)

    sample_baseline(train_split, num=1000, seed=args.seed)
    sample_vae('runs/model.pt', n_dist, num=1000, seed=args.seed, device=args.device)


def cmd_evaluate(args):
    import pickle

    dataset = load_mutag()
    train_split, _, _ = train_val_test_split(dataset)
    training_graphs = [pyg_to_nx(d) for d in train_split]

    with open('runs/baseline_samples.pkl', 'rb') as f:
        baseline_samples = pickle.load(f)
    with open('runs/vae_samples.pkl', 'rb') as f:
        vae_samples = pickle.load(f)

    evaluate_all(training_graphs, baseline_samples, vae_samples)

    plot_stat_grid(training_graphs, baseline_samples, vae_samples)


def cmd_all(args):
    cmd_train(args)
    cmd_sample(args)
    cmd_evaluate(args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['train', 'sample', 'evaluate', 'all'])
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--latent-dim', type=int, default=16)
    parser.add_argument('--state-dim', type=int, default=16)
    parser.add_argument('--num-message-passing-rounds', type=int, default=4)
    parser.add_argument('--beta-max', type=float, default=1.0)
    parser.add_argument('--warmup-epochs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dispatch = {'train': cmd_train, 'sample': cmd_sample, 'evaluate': cmd_evaluate, 'all': cmd_all}
    dispatch[args.command](args)


if __name__ == '__main__':
    main()
