import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_softmax
import math

class SparseGraphormerBlock(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1, max_deg=10):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.d_k = dim // heads

        self.qkv = nn.Linear(dim, dim*3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim*4, dim),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.max_deg = max_deg
        self.deg_emb = nn.Embedding(max_deg + 1, heads)

    def forward(self, x, edge_index, degree_bias, spl_bias=None):
        """
        x: [N, dim]
        edge_index: [2, E]  (source, target)
        degree_bias: [N] long, node degrees
        spl_bias: [E, H] or None, shortest path bias per edge per head
        """

        N = x.size(0)
        E = edge_index.size(1)

        # Q,K,V: [N, 3*dim] -> [N, 3, heads, d_k] -> permute to [3, heads, N, d_k]
        qkv = self.qkv(x).view(N, 3, self.heads, self.d_k).permute(1, 2, 0, 3)
        Q, K, V = qkv[0], qkv[1], qkv[2]  # each: [H, N, d_k]

        src, dst = edge_index  # each: [E]

        # 取出对应边的Q,K
        q = Q[:, src, :]   # [H, E, d_k]
        k = K[:, dst, :]   # [H, E, d_k]

        # 计算每条边每个头的点积注意力分数
        attn = (q * k).sum(dim=-1) / math.sqrt(self.d_k)  # [H, E]

        # 加入 degree bias：deg_emb[node degree] -> (N,H) 转成 (H,N)
        deg_b = self.deg_emb(degree_bias.clamp(max=self.max_deg)).permute(1, 0)  # (H, N)

        # 加入边的degree bias： deg_b[src] + deg_b[dst]
        attn = attn + deg_b[:, src] + deg_b[:, dst]  # [H, E]

        # 加入 shortest-path bias（如果有）
        if spl_bias is not None:
            # 假设 spl_bias shape 是 (E, H), 转置后 (H, E)
            attn = attn + spl_bias.t()

        # 按 source 节点对出边 softmax
        attn = scatter_softmax(attn, src, dim=1)  # 按第二维E维度（即每条边），src是维度1上分组索引

        attn = self.attn_drop(attn)  # dropout

        # 计算输出，加权求和 V[dst]
        v = V[:, dst, :]  # [H, E, d_k]
        out = attn.unsqueeze(-1) * v  # [H, E, d_k]

        # 按 source 聚合到节点上
        out = scatter_add(out, src, dim=1, dim_size=N)  # [H, N, d_k]

        # 转置 reshape
        out = out.permute(1, 0, 2).contiguous().view(N, self.dim)  # [N, dim]

        out = self.out_proj(out)
        out = self.proj_drop(out)

        # 残差 + LayerNorm
        x = self.norm1(x + out)

        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x

class GraphTrans(nn.Module):
    def __init__(self, features, adj, nfeat, nhid, nclass, dropout, heads, device, spl_bias=None):
        super().__init__()
        self.device = device
        self.features = features.to(device)
        self.dropout = dropout

        self.input_proj = nn.Linear(nfeat, nhid)
        self.block = SparseGraphormerBlock(nhid, heads, dropout)
        self.out_proj = nn.Linear(nhid, nclass)

        # edge_index
        edge_index = adj.nonzero(as_tuple=False).t().to(device)  # [2, E]
        self.register_buffer('edge_index', edge_index)

        deg = adj.sum(dim=1).long()
        self.register_buffer('deg', deg)

        # spl_bias: 需要转换成 [E, H] 格式，如果传入的是 (H,N,N) 张量，需要先转换成边对应的格式
        if spl_bias is None:
            self.spl_bias = None
        else:
            spl_bias = spl_bias.to(device)
            # 从 (H,N,N) 转为 (E,H)
            H, N1, N2 = spl_bias.shape
            assert N1 == adj.size(0) and N2 == adj.size(1)
            e_idx = self.edge_index  # [2, E]
            spl_bias_edges = spl_bias[:, e_idx[0], e_idx[1]].permute(1, 0)  # (E,H)
            self.register_buffer('spl_bias', spl_bias_edges)

    def forward(self, nodes):
        x = F.dropout(self.features, self.dropout, training=self.training)
        x = self.input_proj(x)
        x = self.block(x, self.edge_index, self.deg, self.spl_bias)
        x = self.out_proj(x)
        return x[nodes]





class ImprovedFCStacked(nn.Module):
    def __init__(self, input_dim, layer_dims, output_dim, features, device):
        super(ImprovedFCStacked, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.x = features.to(device)
        self.device = device

        self.layers = nn.ModuleList()
        in_dim = input_dim
        for dim in layer_dims:
            self.layers.append(nn.Linear(in_dim, dim))
            in_dim = dim

        self.final_fc = nn.Linear(in_dim, output_dim)
        self.dropout = nn.Dropout(0.5)

        if input_dim != output_dim:
            self.residual_project = nn.Linear(input_dim, output_dim)
        else:
            self.residual_project = nn.Identity()

    def forward(self, nodes):
        x = self.x[nodes]  # (batch_size, input_dim)
        residual = self.residual_project(x)

        for layer in self.layers:
            x = F.relu(layer(x))
            x = self.dropout(x)

        x = self.final_fc(x)
        return x + residual
