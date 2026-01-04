# models/token_graph_net.py
# -*- coding: utf-8 -*-
import math
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv,
    GATv2Conv,
    LayerNorm,
    GraphNorm,
    GlobalAttention,
    global_mean_pool,
    global_max_pool,
    JumpingKnowledge,
)
from torch_geometric.data import Batch


def _make_mlp(dims: List[int], act=nn.GELU, dropout: float = 0.0, last_bn: bool = False):
    layers = []
    for i in range(len(dims) - 1):
        layers += [nn.Linear(dims[i], dims[i + 1])]
        if i < len(dims) - 2 or last_bn:
            layers += [nn.BatchNorm1d(dims[i + 1])]
        if i < len(dims) - 2:
            layers += [act()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
    return nn.Sequential(*layers)


class EdgeWeightedGCNBlock(nn.Module):
    """GCNConv(带 edge_weight) + 残差 + Norm + GELU + Dropout"""
    def __init__(self, dim: int, dropout: float = 0.1, norm_type: str = "layer"):
        super().__init__()
        self.conv = GCNConv(dim, dim, add_self_loops=False, normalize=True)
        self.dropout = nn.Dropout(dropout)
        if norm_type == "graph":
            self.norm = GraphNorm(dim)
        else:
            self.norm = LayerNorm(dim)

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        h = self.conv(x, edge_index, edge_weight=edge_weight)
        h = self.norm(h, batch) if isinstance(self.norm, GraphNorm) else self.norm(h)
        h = F.gelu(h)
        h = self.dropout(h)
        return x + h  # residual


class EdgeAttrGATBlock(nn.Module):
    """
    GATv2Conv（支持 edge_attr，通过 edge_dim 注入），输入输出同维，残差 + Norm + GELU + Dropout
    当 edge_attr 为空时，退化为普通 GATv2 注意力（不含边特征）
    """
    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1, edge_dim: Optional[int] = 1,
                 norm_type: str = "layer"):
        super().__init__()
        self.heads = heads
        self.edge_dim = edge_dim
        # 如果没有边特征，edge_dim=None
        self.conv = GATv2Conv(
            in_channels=dim,
            out_channels=dim // heads,
            heads=heads,
            add_self_loops=False,
            edge_dim=edge_dim,
            dropout=dropout,
            share_weights=True,
        )
        self.proj = nn.Identity() if dim % heads == 0 else nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        if norm_type == "graph":
            self.norm = GraphNorm(dim)
        else:
            self.norm = LayerNorm(dim)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # 若没有 edge_attr，但 conv 需要 edge_dim，会传 None；GATv2Conv 会自行处理
        h = self.conv(x, edge_index, edge_attr=edge_attr)
        if not isinstance(self.proj, nn.Identity):
            h = self.proj(h)
        h = self.norm(h, batch) if isinstance(self.norm, GraphNorm) else self.norm(h)
        h = F.gelu(h)
        h = self.dropout(h)
        return x + h  # residual


class FusionBlock(nn.Module):
    """
    双路融合：GCN(权重) + GAT(edge_attr)，可选只开其中一路。
    融合方式：加权求和 + 残差（内部各自已有残差，这里再做一次线性融合以适配数值尺度）
    """
    def __init__(self, dim: int, use_gcn: bool = True, use_gat: bool = True,
                 heads: int = 4, dropout: float = 0.1, edge_dim: Optional[int] = 1,
                 norm_type: str = "layer"):
        super().__init__()
        assert use_gcn or use_gat, "At least one of GCN/GAT must be enabled."
        self.use_gcn = use_gcn
        self.use_gat = use_gat
        if use_gcn:
            self.gcn = EdgeWeightedGCNBlock(dim, dropout=dropout, norm_type=norm_type)
        if use_gat:
            self.gat = EdgeAttrGATBlock(dim, heads=heads, dropout=dropout, edge_dim=edge_dim, norm_type=norm_type)
        # 融合权重（可学习）
        n = int(use_gcn) + int(use_gat)
        self.alpha = nn.Parameter(torch.ones(n, dtype=torch.float32))

        self.out_norm = LayerNorm(dim) if norm_type == "layer" else GraphNorm(dim)
        self.out_drop = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_weight=None, edge_attr=None, batch=None):
        outs = []
        if self.use_gcn:
            outs.append(self.gcn(x, edge_index, edge_weight=edge_weight, batch=batch))
        if self.use_gat:
            outs.append(self.gat(x, edge_index, edge_attr=edge_attr, batch=batch))
        # 归一化融合权重
        w = torch.softmax(self.alpha, dim=0)
        h = 0
        for i, o in enumerate(outs):
            h = h + w[i] * o
        h = self.out_norm(h, batch) if isinstance(self.out_norm, GraphNorm) else self.out_norm(h)
        h = F.gelu(h)
        h = self.out_drop(h)
        return h


class TokenGraphNet(nn.Module):
    """
    适配你的图数据的主模型：
    - 输入：Data.x [num_nodes, in_dim]；Data.edge_index；可选 Data.edge_weight（标量）；可选 Data.edge_attr（标量/向量）
    - 结构：输入线性编码 -> 多层 FusionBlock(GCN/GAT) -> (可选 JK) -> mean/max/attention 三路读出 -> MLP 分类
    - 输出：logits [batch_size, 1]
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        use_gcn: bool = True,
        use_gat: bool = True,
        gat_heads: int = 4,
        readout_attn_dim: int = 128,
        dropout: float = 0.2,
        jk_mode: Optional[str] = None,  # 'last' | 'max' | 'cat' | None
        norm_type: str = "layer",       # 'layer' or 'graph'
        edge_attr_dim: Optional[int] = 1,  # 你的构图里 edge_attr=scalar -> 1；若没有则传 None
        num_classes: int = 1,           # 二分类 => 1（BCEWithLogits）
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_gcn = use_gcn
        self.use_gat = use_gat
        self.jk_mode = jk_mode

        # 输入投影
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 主干层
        self.layers = nn.ModuleList([
            FusionBlock(
                dim=hidden_dim,
                use_gcn=use_gcn,
                use_gat=use_gat,
                heads=gat_heads,
                dropout=dropout,
                edge_dim=edge_attr_dim,
                norm_type=norm_type,
            ) for _ in range(num_layers)
        ])

        # Jumping Knowledge（可选）
        if jk_mode is not None:
            if jk_mode == "cat":
                self.jk = JumpingKnowledge(mode="cat")
                jk_out_dim = hidden_dim * num_layers
            else:
                self.jk = JumpingKnowledge(mode=jk_mode)  # 'max' or 'last'
                jk_out_dim = hidden_dim
            self.proj_after_jk = nn.Linear(jk_out_dim, hidden_dim)
        else:
            self.jk = None

        # 读出：mean + max + attention
        gate_nn = _make_mlp([hidden_dim, readout_attn_dim, 1], act=nn.GELU, dropout=dropout)
        self.attn_pool = GlobalAttention(gate_nn=gate_nn)

        self.readout_proj = nn.Linear(hidden_dim * 3, hidden_dim)

        # 分类头
        self.cls = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)  # logits
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, data: Batch):
        """
        data: PyG Batch；需要包含
          - x: [N, in_dim]
          - edge_index: [2, E]
          - 可选 edge_weight: [E]
          - 可选 edge_attr: [E, edge_attr_dim] 或 [E]（会自动扩展为 [E,1]）
          - batch: [N] 图索引
        """
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, "batch") else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        edge_weight = getattr(data, "edge_weight", None)
        edge_attr = getattr(data, "edge_attr", None)

        # 兼容 scalar edge_attr
        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.view(-1, 1)

        h = self.enc(x)
        h_list = []
        for layer in self.layers:
            h = layer(h, edge_index, edge_weight=edge_weight, edge_attr=edge_attr, batch=batch)
            if self.jk is not None:
                h_list.append(h)

        if self.jk is not None:
            h = self.jk(h_list)
            if isinstance(self.jk, JumpingKnowledge) and self.jk_mode == "cat":
                h = self.proj_after_jk(h)
            elif isinstance(self.jk, JumpingKnowledge) and self.jk_mode in ("max", "last"):
                h = self.proj_after_jk(h)

        # 读出
        hg_mean = global_mean_pool(h, batch)
        hg_max = global_max_pool(h, batch)
        hg_attn = self.attn_pool(h, batch)
        hg = torch.cat([hg_mean, hg_max, hg_attn], dim=-1)
        hg = self.readout_proj(hg)

        logits = self.cls(hg).view(-1, 1)  # [B, 1]
        return logits

    @torch.no_grad()
    def predict_proba(self, data: Batch):
        logits = self.forward(data)
        return torch.sigmoid(logits)

    def num_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
