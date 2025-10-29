import torch, torch.nn as nn, torch.nn.functional as F

class WindowEncoder(nn.Module):
    def __init__(self, d, m, d_proj=256):
        super().__init__()
        self.proj = nn.Linear(m * d, d_proj)
    def forward(self, win):
        return F.normalize(self.proj(win.reshape(win.size(0), -1)), dim=-1)

class InfoNCE(nn.Module):
    def __init__(self, d_proj=256, temperature=0.07):
        super().__init__()
        self.q = nn.Linear(d_proj, d_proj)
        self.k = nn.Linear(d_proj, d_proj)
        self.temp = nn.Parameter(torch.tensor(temperature))
    def forward(self, qv, kv):
        qn = F.normalize(self.q(qv), dim=-1)
        kn = F.normalize(self.k(kv), dim=-1)
        logits = qn @ kn.t() / self.temp.clamp_min(1e-3)
        labels = torch.arange(qn.size(0), device=qn.device)
        loss = F.cross_entropy(logits, labels)
        return loss, logits
