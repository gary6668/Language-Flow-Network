import torch
import torch.nn as nn
import torch.nn.functional as F

class WindowEncoder(nn.Module):
    """
    输入:
      - win: (B, m, D)  或 (B, m, L, D) 都支持
    策略:
      - 若给 (B,m,D): 自动添加 L=1
      - 线性层输入维度自适配 (lazy init)
    """
    def __init__(self, d=768, m=3, d_proj=128):
        super().__init__()
        self.d = d
        self.m = m
        self.d_proj = d_proj
        self.linear = None  # lazy init

    def forward(self, win):
        if win.ndim == 3:  # (B,m,D) -> (B,m,1,D)
            win = win.unsqueeze(2)

        B, m, L, D = win.shape
        flat_dim = m * L * D
        if self.linear is None or self.linear.in_features != flat_dim:
            self.linear = nn.Linear(flat_dim, self.d_proj, bias=True).to(win.device)

        flat = win.reshape(B, flat_dim)
        out = self.linear(flat)
        # 数值稳定归一化
        norm = torch.norm(out, dim=-1, keepdim=True).clamp_min(1e-8)
        return out / norm


class InfoNCE(nn.Module):
    def __init__(self, d_proj=128, temperature=0.07):
        super().__init__()
        self.q = nn.Linear(d_proj, d_proj)
        self.k = nn.Linear(d_proj, d_proj)
        self.temp = nn.Parameter(torch.tensor(temperature, dtype=torch.float32))

    def forward(self, qv, kv):
        eps = 1e-8
        qn = F.normalize(self.q(qv), dim=-1, eps=eps)
        kn = F.normalize(self.k(kv), dim=-1, eps=eps)
        logits = (qn @ kn.t()) / self.temp.clamp_min(0.01)
        if torch.isnan(logits).any():
            return torch.tensor(float('nan'), device=logits.device), logits
        labels = torch.arange(qn.size(0), device=qn.device)
        loss = F.cross_entropy(logits, labels)
        return loss, logits