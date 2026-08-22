"""Euler LANL encoders (GCN / GAT / GraphSAGE) + LinkMLP link decoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv, GCNConv
from torch_geometric.nn.conv.message_passing import MessagePassing


class DropEdge(nn.Module):
    """Euler DropEdge (p=0.8) before GNN — https://openreview.net/forum?id=Hkx1qkrKPr"""

    def __init__(self, p: float = 0.8):
        super().__init__()
        self.p = p

    def forward(self, ei: torch.Tensor, ew: torch.Tensor | None = None):
        if self.training and self.p > 0:
            mask = torch.rand(ei.size(1), device=ei.device) > self.p
            ei = ei[:, mask]
            if ew is not None:
                ew = ew[mask]
        if ew is None:
            return ei
        return ei, ew


class PoolSAGEConv(MessagePassing):
    """Euler max-pool GraphSAGE (matches embedders.py)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr="max")
        self.aggr_n = nn.Sequential(nn.Linear(in_channels, out_channels), nn.ReLU())
        self.e_lin = nn.Linear(out_channels, out_channels)
        self.r_lin = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x_e = self.aggr_n(x)
        x_e = self.propagate(edge_index, x=x_e, size=None)
        x_e = self.e_lin(x_e)
        x_r = self.r_lin(x)
        return F.normalize(x_r + x_e, p=2.0, dim=-1)


class _EulerEncoderBase(nn.Module):
    """Shared: DropEdge → 2-layer encoder → tanh (Euler embedders.py)."""

    def __init__(self, drop_edge_p: float = 0.8, dropout_p: float = 0.25):
        super().__init__()
        self.drop_edge = DropEdge(drop_edge_p)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p)
        self.tanh = nn.Tanh()


class GCNEuler(_EulerEncoderBase):
    def __init__(self, in_dim: int, hidden: int = 32, out_dim: int = 16):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden, add_self_loops=True)
        self.conv2 = GCNConv(hidden, hidden, add_self_loops=True)
        self.conv3 = GCNConv(hidden, out_dim, add_self_loops=True)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor):
        ei, ew = self.drop_edge(edge_index, edge_weight)
        x = self.relu(self.conv1(x, ei, edge_weight=ew))
        x = self.dropout(x)
        x = self.relu(self.conv2(x, ei, edge_weight=ew))
        x = self.dropout(x)
        x = self.conv3(x, ei, edge_weight=ew)
        return self.tanh(x)


class GATEuler(_EulerEncoderBase):
    """GAT with 3 heads, concat=False; no edge weights (Euler default)."""

    def __init__(self, in_dim: int, hidden: int = 32, out_dim: int = 16, heads: int = 3):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden, heads=heads, concat=False)
        self.conv2 = GATConv(hidden, out_dim, heads=heads, concat=False)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor | None = None):
        ei = self.drop_edge(edge_index)
        x = self.conv1(x, ei)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, ei)
        return self.tanh(x)


class GATv2Euler(_EulerEncoderBase):
    """GATv2 with optional multi-dim edge_attr (full CARS upgrade path)."""

    def __init__(
        self,
        in_dim: int,
        hidden: int = 32,
        out_dim: int = 16,
        heads: int = 3,
        edge_dim: int = 7,
    ):
        super().__init__()
        self.conv1 = GATv2Conv(in_dim, hidden, heads=heads, concat=False, edge_dim=edge_dim)
        self.conv2 = GATv2Conv(hidden, out_dim, heads=heads, concat=False, edge_dim=edge_dim)

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ):
        ei = self.drop_edge(edge_index)
        ea = edge_attr
        x = self.conv1(x, ei, edge_attr=ea)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, ei, edge_attr=ea)
        return self.tanh(x)


class SAGEEuler(_EulerEncoderBase):
    def __init__(self, in_dim: int, hidden: int = 32, out_dim: int = 16):
        super().__init__()
        self.conv1 = PoolSAGEConv(in_dim, hidden)
        self.conv2 = PoolSAGEConv(hidden, out_dim)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor):
        ei = self.drop_edge(edge_index)
        x = self.conv1(x, ei)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, ei)
        return self.tanh(x)


class LinkMLP(nn.Module):
    """Unsupervised link decoder: [h_u, h_v, edge_attr] (+ optional dot-product)."""

    def __init__(
        self,
        embed_dim: int = 16,
        edge_feat_dim: int = 7,
        include_dot: bool = False,
        hidden: int = 64,
        dropout: float = 0.25,
    ):
        super().__init__()
        in_dim = 2 * embed_dim + edge_feat_dim + (1 if include_dot else 0)
        self.include_dot = include_dot
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        h_u: torch.Tensor,
        h_v: torch.Tensor,
        edge_attr: torch.Tensor,
        link_logit: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts = [h_u, h_v, edge_attr]
        if self.include_dot:
            if link_logit is None:
                link_logit = (h_u * h_v).sum(dim=-1)
            parts.append(link_logit.unsqueeze(-1))
        return self.net(torch.cat(parts, dim=-1)).squeeze(-1)


def build_model(
    name: str,
    in_dim: int,
    *,
    gat_variant: str = "v1",
    edge_feat_dim: int = 0,
) -> nn.Module:
    if name == "gat" and gat_variant == "v2":
        if edge_feat_dim <= 0:
            raise ValueError("gat v2 requires edge_feat_dim > 0")
        return GATv2Euler(in_dim, edge_dim=edge_feat_dim)
    builders = {
        "gcn": lambda: GCNEuler(in_dim),
        "gat": lambda: GATEuler(in_dim),
        "sage": lambda: SAGEEuler(in_dim),
    }
    return builders[name]()


def build_link_mlp(
    edge_feat_dim: int = 7,
    embed_dim: int = 16,
    include_dot: bool = False,
) -> LinkMLP:
    return LinkMLP(
        embed_dim=embed_dim,
        edge_feat_dim=edge_feat_dim,
        include_dot=include_dot,
    )
