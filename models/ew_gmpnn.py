# models/ew_gmpnn.py
# -*- coding: utf-8 -*-
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn import LayerNorm, GraphNorm, GlobalAttention, global_mean_pool, global_max_pool
from torch_geometric.data import Batch


def _mlp(dims, act=nn.GELU, dropout=0.0, bn_last=False):
    layers = []
    for i in range(len(dims) - 1):
        layers += [nn.Linear(dims[i], dims[i+1])]
        if i < len(dims) - 2 or bn_last:
            layers += [nn.BatchNorm1d(dims[i+1])]
        if i < len(dims) - 2:
            layers += [act()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
    return nn.Sequential(*layers)


def _weighted_degree(num_nodes: int, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
    # out-degree; 你的图已是双向边，out/in 基本等价
    deg = scatter_add(edge_weight, edge_index[0], dim=0, dim_size=num_nodes)
    return deg.clamp(min=1e-8)


def _norm_weight(num_nodes: int, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
    deg = _weighted_degree(num_nodes, edge_index, edge_weight)
    di = deg[edge_index[0]]
    dj = deg[edge_index[1]]
    return edge_weight / (di.sqrt() * dj.sqrt() + 1e-8), deg


def _edge_feats(edge_weight_norm: torch.Tensor) -> torch.Tensor:
    # [E, 2]: [w_norm, z_score(w_norm)]
    w = edge_weight_norm
    mu = w.mean()
    std = w.std(unbiased=False).clamp(min=1e-8)
    z = (w - mu) / std
    return torch.stack([w, z], dim=-1)


class EdgeGatedMPNNLayer(nn.Module):
    """
    m_ij = gate([x_i, x_j, e]) * w_norm * phi(x_j)
    h_i' = Norm( h_i + Drop(GELU( MLP([h_i, sum_j m_ij]) )) )
    """
    def __init__(self, dim: int, edge_feat_dim: int = 2, dropout: float = 0.1, norm_type: str = "layer"):
        super().__init__()
        self.phi = nn.Linear(dim, dim)  # message projection
        self.edge_gate = _mlp([dim*2 + edge_feat_dim, dim, 1], dropout=dropout)
        self.upd = _mlp([dim*2, dim], dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.norm = LayerNorm(dim) if norm_type == "layer" else GraphNorm(dim)

    def forward(self, x, edge_index, edge_weight_norm, edge_feat, batch):
        src, dst = edge_index
        xj = self.phi(x)  # [N, D]
        # gate in (0,1)
        g_in = torch.cat([x[dst], x[src], edge_feat], dim=-1)  # 注意顺序：i接收，concat[x_i, x_j, e]
        gate = torch.sigmoid(self.edge_gate(g_in)).view(-1)    # [E]
        msg = xj[src] * (edge_weight_norm * gate).unsqueeze(-1)  # [E, D]
        agg = scatter_add(msg, dst, dim=0, dim_size=x.size(0))   # [N, D]

        h = torch.cat([x, agg], dim=-1)
        h = self.upd(h)
        h = F.gelu(h)
        h = self.drop(h)
        h = self.norm(x + h, batch) if isinstance(self.norm, GraphNorm) else self.norm(x + h)
        return h


class PPRDiffuse(nn.Module):
    """
    固定图先验扩散残差：K 步 personalized propagation
    H^{t+1} = (1 - alpha) * A_norm @ H^t + alpha * H^0
    """
    def __init__(self, alpha: float = 0.1, K: int = 5):
        super().__init__()
        self.alpha = alpha
        self.K = K
        self.scale = nn.Parameter(torch.tensor(0.5))  # 学习一个残差比例

    def forward(self, h0, edge_index, edge_weight_norm):
        h = h0
        src, dst = edge_index
        for _ in range(self.K):
            msg = h[src] * edge_weight_norm.unsqueeze(-1)
            h = scatter_add(msg, dst, dim=0, dim_size=h0.size(0))
            h = (1 - self.alpha) * h + self.alpha * h0
        return h0 + torch.tanh(self.scale) * h  # 限幅，避免爆炸


class StructEncoder(nn.Module):
    """在线构造结构特征：加权度、log度、简易PR（几步 power-iteration），然后映射到 dim 并加到节点表示。"""
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(3, dim)

    def forward(self, x, edge_index, edge_weight_norm):
        N = x.size(0)
        deg = scatter_add(edge_weight_norm, edge_index[0], dim=0, dim_size=N).clamp(min=1e-8)
        logdeg = deg.log()
        # 简易 PR：几步迭代（无 teleport，为避免重复引入 alpha，这里只做 3 步归一传播）
        pr = torch.full((N,), 1.0 / N, device=x.device, dtype=x.dtype)
        src, dst = edge_index
        for _ in range(3):
            msg = pr[src] * edge_weight_norm
            pr = scatter_add(msg, dst, dim=0, dim_size=N)
            pr = pr / pr.sum().clamp(min=1e-8)

        s = torch.stack([deg, logdeg, pr], dim=-1)
        s = (s - s.mean(0, keepdim=True)) / (s.std(0, keepdim=True).clamp(min=1e-8))
        return self.proj(s)  # [N, D]


class EWGMPNN(nn.Module):
    """
    Edge-Weight Gated MPNN + PPR残差 + 结构特征
    适配：Data.x, Data.edge_index, Data.edge_weight(必需), 可选 Data.batch
    输出 logits [B,1]
    """
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int = 256,
                 num_layers: int = 4,
                 dropout: float = 0.2,
                 norm_type: str = "layer",
                 use_ppr_residual: bool = True,
                 ppr_alpha: float = 0.1,
                 ppr_steps: int = 5):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.struct_enc = StructEncoder(hidden_dim)

        self.layers = nn.ModuleList([
            EdgeGatedMPNNLayer(hidden_dim, edge_feat_dim=2, dropout=dropout, norm_type=norm_type)
            for _ in range(num_layers)
        ])

        self.use_ppr = use_ppr_residual
        self.ppr = PPRDiffuse(alpha=ppr_alpha, K=ppr_steps) if use_ppr_residual else None

        gate_nn = _mlp([hidden_dim, 128, 1], dropout=dropout)
        self.attn_pool = GlobalAttention(gate_nn=gate_nn)
        self.readout = nn.Linear(hidden_dim * 3, hidden_dim)

        self.cls = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self._reset()

    def _reset(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, data: Batch):
        x = data.x
        edge_index = data.edge_index
        edge_weight = getattr(data, "edge_weight", None)
        if edge_weight is None:
            raise ValueError("EWGMPNN 需要 Data.edge_weight（你的构图权重）。")
        batch = getattr(data, "batch", torch.zeros(x.size(0), dtype=torch.long, device=x.device))

        # 归一化边权 + 边特征
        w_norm, deg = _norm_weight(x.size(0), edge_index, edge_weight)
        e_feat = _edge_feats(w_norm)  # [E,2]

        # 编码 + 结构特征注入
        h = self.enc(x) + self.struct_enc(x, edge_index, w_norm)

        for layer in self.layers:
            h = layer(h, edge_index, w_norm, e_feat, batch)

        # 可选：PPR 残差（固定扩散）
        if self.use_ppr:
            h = self.ppr(h, edge_index, w_norm)

        # 读出：mean / max / attention
        hg_mean = global_mean_pool(h, batch)
        hg_max  = global_max_pool(h, batch)
        hg_attn = self.attn_pool(h, batch)
        hg = torch.cat([hg_mean, hg_max, hg_attn], dim=-1)
        hg = self.readout(hg)

        logits = self.cls(hg).view(-1, 1)
        return logits

    @torch.no_grad()
    def predict_proba(self, data: Batch):
        return torch.sigmoid(self.forward(data))
