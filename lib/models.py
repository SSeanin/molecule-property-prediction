import torch
from torch.nn import Module, Linear
from torch_geometric.nn import global_mean_pool

from .layers import MPNNLayer


class MPNNModel(Module):
    def __init__(self, num_layers=4, emb_dim=64, in_dim=11, edge_dim=4, out_dim=1):
        super().__init__()

        self.lin_in = Linear(in_dim, emb_dim)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(MPNNLayer(emb_dim, edge_dim, aggr='add'))

        self.pool = global_mean_pool

        self.lin_pred = Linear(emb_dim, out_dim)

    def forward(self, data):
        h = self.lin_in(data.x)

        for conv in self.convs:
            h = h + conv(h, data.edge_index, data.edge_attr)

        h_graph = self.pool(h, data.batch)

        out = self.lin_pred(h_graph)

        return out.view(-1)
