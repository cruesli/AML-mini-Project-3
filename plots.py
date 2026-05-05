import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from evaluate import graph_stats


def plot_loss_curve(history, path='runs/loss_curve.png'):
    os.makedirs('runs', exist_ok=True)
    epochs = sorted(history.keys())[1:]  # skip epoch 0 (extreme init values)
    recons = [history[e]['recon'] for e in epochs]
    kls = [history[e]['kl'] for e in epochs]
    val_recons = [history[e]['val_recon'] for e in epochs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(epochs, recons, label='train')
    ax1.plot(epochs, val_recons, label='val', linestyle='--')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Reconstruction')
    ax1.legend()
    ax1.spines[['top', 'right']].set_visible(False)

    ax2.plot(epochs, kls)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('KL')
    ax2.spines[['top', 'right']].set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_loss_csv(csv_path='runs/training_log.csv', path='runs/loss_curve_csv.png', skip_epochs=2):
    import csv
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items()})
    rows = rows[skip_epochs:]

    epochs      = [r['epoch']      for r in rows]
    train_recon = [r['train_recon'] for r in rows]
    val_recon   = [r['val_recon']   for r in rows]
    train_kl    = [r['train_kl']    for r in rows]
    val_kl      = [r['val_kl']      for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(epochs, train_recon, label='train')
    ax1.plot(epochs, val_recon,   label='val', linestyle='--')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Reconstruction')
    ax1.legend()
    ax1.spines[['top', 'right']].set_visible(False)

    ax2.plot(epochs, train_kl, label='train')
    ax2.plot(epochs, val_kl,   label='val', linestyle='--')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('KL')
    ax2.legend()
    ax2.spines[['top', 'right']].set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'Saved {path}')


def _collect_stats(graphs):
    degrees, clustering, eigvec = [], [], []
    for G in graphs:
        d, c, e = graph_stats(G)
        degrees.extend(d)
        clustering.extend(c)
        eigvec.extend(e)
    return degrees, clustering, eigvec


def plot_stat_grid(training_graphs, baseline_graphs, vae_graphs, path='runs/stat_grid.png'):
    os.makedirs('runs', exist_ok=True)

    # palette: neutral grey for empirical, two accent colors
    colors = ['#888888', '#e07b39', '#4472c4']
    row_labels = ['Empirical', 'Erdős–Rényi', 'GraphVAE']
    col_labels = ['Node degree', 'Clustering coefficient', 'Eigenvector centrality']

    sources = [training_graphs, baseline_graphs, vae_graphs]
    all_stats = [_collect_stats(gs) for gs in sources]  # list of (deg, clust, eig)

    fig, axes = plt.subplots(3, 3, figsize=(10, 8), sharey=False)

    for col in range(3):
        # shared bins computed jointly across all three sources
        combined = []
        for stats in all_stats:
            combined.extend(stats[col])
        combined = np.array(combined, dtype=float)
        if len(combined) == 0 or np.all(combined == combined[0]):
            bins = np.linspace(0, 1, 31)
        else:
            bins = np.histogram_bin_edges(combined, bins=30)

        for row in range(3):
            ax = axes[row][col]
            vals = np.array(all_stats[row][col], dtype=float)
            if len(vals) > 0:
                ax.hist(vals, bins=bins, density=True, color=colors[row], alpha=0.8, edgecolor='none')
            ax.spines[['top', 'right']].set_visible(False)

            # axis labels only on outer cells
            if row == 2:
                ax.set_xlabel(col_labels[col])
            if col == 0:
                ax.set_ylabel('Density')

    # row labels on left of leftmost column
    for row, label in enumerate(row_labels):
        axes[row][0].annotate(
            label, xy=(0, 0.5), xycoords='axes fraction',
            xytext=(-0.45, 0.5), textcoords='axes fraction',
            ha='right', va='center', fontsize=10, rotation=0,
        )

    # column labels on top row
    for col, label in enumerate(col_labels):
        axes[0][col].set_title(label)

    fig.tight_layout()
    fig.subplots_adjust(left=0.18)
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    plot_loss_csv()
