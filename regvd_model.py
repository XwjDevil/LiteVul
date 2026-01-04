import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, JumpingKnowledge
from torch_geometric.nn import global_add_pool, global_max_pool
from torch_scatter import scatter_add
from torch_geometric.nn import GlobalAttention

class SimpleEdgeAwareUniNet(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=256, num_classes=2, num_layers=3, heads=4, use_degree_gate=True):
        super().__init__()
        self.num_layers = num_layers
        self.heads = heads
        self.use_degree_gate = use_degree_gate

        # 输入投影层
        self.in_proj = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()
        
        # TransformerConv 层
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            out_channels = hidden_dim // heads
            self.convs.append(
                TransformerConv(in_channels=hidden_dim, out_channels=out_channels, heads=heads, dropout=0.0, edge_dim=1, beta=True, bias=True)
            )
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        # 跨层聚合：JumpingKnowledge
        self.jk = JumpingKnowledge(mode='max')

        # 度感知注意力池化
        gate_in = hidden_dim + (2 if use_degree_gate else 0)
        self.gate_nn = nn.Sequential(
            nn.Linear(gate_in, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.global_att = None if use_degree_gate else GlobalAttention(self.gate_nn)
        readout_dim = hidden_dim * 3
        self.cls = nn.Sequential(
            nn.Linear(readout_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    @staticmethod
    def _prepare_edge_attr(edge_index, edge_attr):
        if edge_attr is None:
            return None
        e = edge_attr.view(-1)
        e = torch.log1p(e)
        from torch_geometric.utils import softmax
        alpha = softmax(e, edge_index[0])
        return alpha.view(-1, 1)

    @staticmethod
    def _degree_features(num_nodes, edge_index, edge_attr_ew):
        w = edge_attr_ew.view(-1)
        out_w = scatter_add(w, edge_index[0], dim=0, dim_size=num_nodes)
        in_w = scatter_add(w, edge_index[1], dim=0, dim_size=num_nodes)
        return torch.stack([torch.log1p(out_w), torch.log1p(in_w)], dim=1)

    def _degree_aware_attention_pool(self, x, deg_feat, batch):
        z = torch.cat([x, deg_feat], dim=1)
        gate = torch.sigmoid(self.gate_nn(z))
        return global_add_pool(x * gate, batch)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        edge_attr = self._prepare_edge_attr(edge_index, edge_attr)

        h = self.in_proj(x)
        h_list = []
        for i in range(self.num_layers):
            h_norm = self.norms[i](h)
            h_new = self.convs[i](h_norm, edge_index, edge_attr=edge_attr)
            h_new = F.gelu(h_new)
            h = h + h_new
            h_list.append(h)
        #h_jk = self.jk(h_list)
        h_jk = torch.mean(torch.stack(h_list, dim=0), dim=0)


        if self.use_degree_gate and edge_attr is not None:
            deg_feat = self._degree_features(h_jk.size(0), edge_index, edge_attr.view(-1))
            attn_pool = self._degree_aware_attention_pool(h_jk, deg_feat, batch)
        else:
            attn_pool = (self.global_att or GlobalAttention(self.gate_nn))(h_jk, batch)

        sum_pool = global_add_pool(h_jk, batch)
        max_pool = global_max_pool(h_jk, batch)
        graph_emb = torch.cat([attn_pool, sum_pool, max_pool], dim=1)

        return self.cls(graph_emb)

# #uni+gcn
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv, global_max_pool, global_add_pool

# class ResidualGNNLayer(nn.Module):
#     def __init__(self, in_dim, out_dim):
#         super().__init__()
#         self.gnn = GCNConv(in_dim, out_dim)
#         self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else None

#     def forward(self, x, edge_index):
#         h_new = self.gnn(x, edge_index)
#         if self.proj is not None:
#             x = self.proj(x)
#         return x + h_new

# class ReGVDModel(nn.Module):
#     def __init__(self, in_dim, hidden_dim, num_classes, num_layers=2, readout='concat', dropout=0.5):
#         """
#         in_dim: 节点输入特征维度
#         hidden_dim: GNN隐藏层维度
#         num_classes: 分类类别数
#         num_layers: GNN层数
#         readout: 'concat'/'sum'/'mul'
#         dropout: Dropout概率
#         """
#         super().__init__()
#         self.layers = nn.ModuleList()
#         self.layers.append(ResidualGNNLayer(in_dim, hidden_dim))
#         for _ in range(num_layers-1):
#             self.layers.append(ResidualGNNLayer(hidden_dim, hidden_dim))
#         self.readout = readout
#         out_dim = hidden_dim * 2 if readout == 'concat' else hidden_dim
#         self.dropout = nn.Dropout(dropout)
#         self.classifier = nn.Sequential(
#             nn.Linear(out_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, num_classes)
#         )

#     def forward(self, data):
#         x, edge_index, batch = data.x, data.edge_index, data.batch
#         h = x
#         for layer in self.layers:
#             h = layer(h, edge_index)
#             h = F.relu(h)
#             h = self.dropout(h)   # <--- GNN层输出后加Dropout
#         # Graph-level readout
#         sum_pool = global_add_pool(h, batch)
#         max_pool = global_max_pool(h, batch)
#         if self.readout == 'concat':
#             graph_emb = torch.cat([sum_pool, max_pool], dim=1)
#         elif self.readout == 'sum':
#             graph_emb = sum_pool + max_pool
#         elif self.readout == 'mul':
#             graph_emb = sum_pool * max_pool
#         else:
#             raise ValueError("readout must be 'concat', 'sum', or 'mul'")
#         out = self.classifier(graph_emb)
#         return out


#uni+gat
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GATConv, global_max_pool, global_add_pool

# class ReGVDModelGAT(nn.Module):
#     def __init__(self, in_dim, hidden_dim, num_classes, num_layers=2, readout='mul', dropout=0.5):
#         """
#         in_dim: 节点输入特征维度
#         hidden_dim: GAT隐藏层维度
#         num_classes: 分类类别数
#         num_layers: GAT层数
#         readout: 'concat'/'sum'/'mul'
#         dropout: Dropout概率
#         """
#         super().__init__()
#         self.layers = nn.ModuleList()
#         self.layers.append(GATConv(in_dim, hidden_dim, heads=1, concat=True))  # 第一层
#         for _ in range(num_layers-1):
#             self.layers.append(GATConv(hidden_dim, hidden_dim, heads=1, concat=True))  # 后续层
#         self.readout = readout
#         out_dim = hidden_dim * 2 if readout == 'concat' else hidden_dim
#         self.dropout = nn.Dropout(dropout)
#         self.classifier = nn.Sequential(
#             nn.Linear(out_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, num_classes)
#         )

#     def forward(self, data):
#         """
#         data: PyG Batch对象，含.x, .edge_index, .edge_attr, .batch
#         """
#         x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
#         h = x
#         for layer in self.layers:
#             h = layer(h, edge_index)  # GAT使用自注意力机制来加权邻居节点
#             h = F.relu(h)
#             h = self.dropout(h)
        
#         # Graph-level readout
#         sum_pool = global_add_pool(h, batch)
#         max_pool = global_max_pool(h, batch)
#         if self.readout == 'concat':
#             graph_emb = torch.cat([sum_pool, max_pool], dim=1)
#         elif self.readout == 'sum':
#             graph_emb = sum_pool + max_pool
#         elif self.readout == 'mul':
#             graph_emb = sum_pool * max_pool
#         else:
#             raise ValueError("readout must be 'concat', 'sum', or 'mul'")
#         out = self.classifier(graph_emb)
#         return out



















