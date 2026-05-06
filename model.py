import torch
import torch.nn as nn
import torch.distributions as td
import networkx as nx

from data import N_MAX


# encoder backbone adapted from SimpleGNN
class MPNNEncoderNet(nn.Module):
    def __init__(self, node_feature_dim, state_dim, num_message_passing_rounds, latent_dim):
        super().__init__()
        self.state_dim = state_dim
        self.num_message_passing_rounds = num_message_passing_rounds

        self.input_net = nn.Sequential(
            nn.Linear(node_feature_dim, state_dim),
            nn.ReLU(),
        )
        self.message_net = nn.ModuleList([
            nn.Sequential(nn.Linear(state_dim, state_dim), nn.ReLU())
            for _ in range(num_message_passing_rounds)
        ])
        self.update_net = nn.ModuleList([
            nn.Sequential(nn.Linear(state_dim, state_dim), nn.ReLU())
            for _ in range(num_message_passing_rounds)
        ])
        self.output_net = nn.Linear(state_dim, 2 * latent_dim)

    def forward(self, x, edge_index, batch):
        num_nodes = x.shape[0]
        num_graphs = int(batch.max()) + 1

        state = self.input_net(x)
        for r in range(self.num_message_passing_rounds):
            message = self.message_net[r](state)
            aggregated = x.new_zeros(num_nodes, self.state_dim)
            aggregated = aggregated.index_add(0, edge_index[1], message[edge_index[0]])
            state = state + self.update_net[r](aggregated)

        graph_state = x.new_zeros(num_graphs, self.state_dim)
        graph_state = torch.index_add(graph_state, 0, batch, state)
        return self.output_net(graph_state)


# prior
class GaussianPrior(nn.Module):
    def __init__(self, M):
        super().__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(M), requires_grad=False)

    def forward(self):
        return td.Independent(td.Normal(self.mean, self.std), 1)


# encoder
class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        super().__init__()
        self.encoder_net = encoder_net

    def forward(self, data):
        out = self.encoder_net(data.x, data.edge_index, data.batch)
        mean, log_std = torch.chunk(out, 2, dim=-1)
        return td.Independent(td.Normal(mean, torch.exp(log_std)), 1)


# decoder — no td.Independent so per-edge log_prob stays as a K-vector
class BernoulliEdgeDecoder(nn.Module):
    def __init__(self, latent_dim, n_max=N_MAX):
        super().__init__()
        self.n_max = n_max
        k = n_max * (n_max - 1) // 2
        self.decoder_net = nn.Sequential(
            nn.Linear(latent_dim + n_max + 1, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, k),
        )

    def forward(self, z, n_onehot):
        inp = torch.cat([z, n_onehot], dim=-1)
        return td.Bernoulli(logits=self.decoder_net(inp))


# VAE composition
class GraphVAE(nn.Module):
    def __init__(self, prior, encoder, decoder):
        super().__init__()
        self.prior = prior
        self.encoder = encoder
        self.decoder = decoder

    def elbo(self, data, target, mask):
        q = self.encoder(data)
        z = q.rsample()
        ns = torch.bincount(data.batch, minlength=z.shape[0])
        n_onehot = torch.nn.functional.one_hot(ns, num_classes=N_MAX + 1).float()
        p = self.decoder(z, n_onehot)
        k = N_MAX * (N_MAX - 1) // 2
        b = z.shape[0]
        target = target.view(b, k)
        mask = mask.view(b, k)
        recon = (p.log_prob(target) * mask).sum(dim=-1).mean()
        kl = td.kl_divergence(q, self.prior()).mean()
        return recon, kl

    def forward(self, data, target, mask):
        recon, kl = self.elbo(data, target, mask)
        return -(recon - kl)

    def sample(self, num_graphs, n_dist, device='cpu'):
        rows, cols = torch.triu_indices(N_MAX, N_MAX, offset=1)
        ns = torch.multinomial(
            torch.tensor(n_dist, dtype=torch.float), num_graphs, replacement=True
        )
        with torch.no_grad():
            z = self.prior().sample((num_graphs,)).to(device)
            n_onehot = torch.nn.functional.one_hot(ns, num_classes=N_MAX + 1).float().to(device)
            logits = self.decoder(z, n_onehot).logits
            edges = torch.bernoulli(torch.sigmoid(logits)).cpu()

        graphs = []
        for i, n in enumerate(ns.tolist()):
            A = torch.zeros(N_MAX, N_MAX)
            A[rows, cols] = edges[i]
            A = A + A.t()
            G = nx.Graph()
            G.add_nodes_from(range(n))
            block = A[:n, :n]
            for u in range(n):
                for v in range(u + 1, n):
                    if block[u, v].item() > 0.5:
                        G.add_edge(u, v)
            graphs.append(G)
        return graphs


def build_model(latent_dim=16, state_dim=16, num_message_passing_rounds=4,
                node_feature_dim=7, n_max=N_MAX):
    prior = GaussianPrior(latent_dim)
    encoder_net = MPNNEncoderNet(node_feature_dim, state_dim, num_message_passing_rounds, latent_dim)
    encoder = GaussianEncoder(encoder_net)
    decoder = BernoulliEdgeDecoder(latent_dim, n_max)
    return GraphVAE(prior, encoder, decoder)


if __name__ == '__main__':
    from torch_geometric.loader import DataLoader
    from data import load_mutag, train_val_test_split, attach_targets

    dataset = load_mutag()
    train, _, _ = train_val_test_split(dataset)
    train_data = attach_targets(train)
    loader = DataLoader(train_data, batch_size=100)
    batch = next(iter(loader))

    model = build_model()
    recon, kl = model.elbo(batch, batch.A_target, batch.A_mask)
    print(f'recon : {recon.item():.4f}')
    print(f'kl    : {kl.item():.4f}')
    print(f'finite: recon={torch.isfinite(recon).item()}, kl={torch.isfinite(kl).item()}')
    print(f'kl > 0: {kl.item() > 0}')
