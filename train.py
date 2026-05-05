import os
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from data import load_mutag, train_val_test_split, attach_targets
from model import build_model


def eval_elbo(model, loader, device):
    model.eval()
    total_recon = total_kl = n = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            recon, kl = model.elbo(batch, batch.A_target, batch.A_mask)
            total_recon += recon.item()
            total_kl += kl.item()
            n += 1
    model.train()
    return total_recon / max(n, 1), total_kl / max(n, 1)


def train(model, optimizer, scheduler, train_loader, val_loader,
          epochs, beta_max, warmup_epochs, device):
    os.makedirs('runs', exist_ok=True)
    model.train()

    total_steps = len(train_loader) * epochs
    progress = tqdm(range(total_steps), desc='Training')
    history = {}
    best_val_elbo = float('-inf')

    log_path = 'runs/training_log.csv'
    with open(log_path, 'w') as f:
        f.write('epoch,beta,train_recon,train_kl,val_recon,val_kl\n')

    for epoch in range(epochs):
        beta = beta_max * min(1.0, epoch / max(warmup_epochs, 1))
        epoch_recon = epoch_kl = 0.0
        n_batches = 0

        for batch in train_loader:
            batch = batch.to(device)
            recon, kl = model.elbo(batch, batch.A_target, batch.A_mask)
            loss = -(recon - beta * kl)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_recon += recon.item()
            epoch_kl += kl.item()
            n_batches += 1

            progress.set_postfix(
                epoch=f'{epoch+1}/{epochs}',
                recon=f'{recon.item():.2f}',
                kl=f'{kl.item():.2f}',
                beta=f'{beta:.3f}',
            )
            progress.update()

        scheduler.step()

        val_recon, val_kl = eval_elbo(model, val_loader, device)
        val_elbo = val_recon - val_kl
        if val_elbo > best_val_elbo:
            best_val_elbo = val_elbo
            torch.save(model.state_dict(), 'runs/model.pt')

        train_recon = epoch_recon / n_batches
        train_kl = epoch_kl / n_batches
        history[epoch] = {
            'recon': train_recon,
            'kl': train_kl,
            'val_recon': val_recon,
            'val_kl': val_kl,
        }
        with open(log_path, 'a') as f:
            f.write(f'{epoch+1},{beta:.4f},{train_recon:.4f},{train_kl:.4f},{val_recon:.4f},{val_kl:.4f}\n')

    return history


def plot_loss_curve(history, path='runs/loss_curve.png'):
    epochs = sorted(history.keys())
    recons = [history[e]['recon'] for e in epochs]
    kls = [history[e]['kl'] for e in epochs]
    val_recons = [history[e]['val_recon'] for e in epochs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(epochs, recons, label='train')
    ax1.plot(epochs, val_recons, label='val')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Recon')
    ax1.legend()
    ax1.spines[['top', 'right']].set_visible(False)

    ax2.plot(epochs, kls)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('KL')
    ax2.spines[['top', 'right']].set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    device = 'cpu'

    dataset = load_mutag()
    train_split, val_split, _ = train_val_test_split(dataset)
    train_data = attach_targets(train_split)
    val_data = attach_targets(val_split)

    train_loader = DataLoader(train_data, batch_size=100, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=44)

    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)

    history = train(model, optimizer, scheduler, train_loader, val_loader,
                    epochs=10, beta_max=1.0, warmup_epochs=50, device=device)

    print('\n--- 10-epoch check ---')
    for e in sorted(history):
        h = history[e]
        print(f'  epoch {e+1:2d}: recon={h["recon"]:.2f}, kl={h["kl"]:.2f}, '
              f'val_recon={h["val_recon"]:.2f}, val_kl={h["val_kl"]:.2f}')
