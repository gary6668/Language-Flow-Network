import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================================
# ✅ WindowEncoder：将时间窗口（多句嵌入）压缩成单个语义向量
# ==========================================================
class WindowEncoder(nn.Module):
    """
    输入：win shape = (batch, window, 768)
    输出：(batch, d_proj)
    """
    def __init__(self, d=768, m=3, d_proj=256):
        super().__init__()
        self.d = d          # 每个句向量维度
        self.m = m          # 窗口大小
        self.d_proj = d_proj
        self.proj = nn.Linear(m * d, d_proj)

    def forward(self, win):
        # 防止输入维度错误
        if win.ndim != 3:
            raise ValueError(f"Expected (B, m, d) tensor, got {win.shape}")
        # 展平后投影
        flat = win.reshape(win.size(0), self.m * self.d)  # (B, m*d)
        out = self.proj(flat)                             # (B, d_proj)
        return F.normalize(out, dim=-1)


# ==========================================================
# ✅ InfoNCE Mutual Information Estimator
# ==========================================================
class InfoNCE(nn.Module):
    """
    qv, kv: shape = (B, d_proj)
    """
    def __init__(self, d_proj=256, temperature=0.07):
        super().__init__()
        self.d_proj = d_proj
        self.q = nn.Linear(d_proj, d_proj)
        self.k = nn.Linear(d_proj, d_proj)
        self.temp = nn.Parameter(torch.tensor(temperature, dtype=torch.float32))

    def forward(self, qv, kv):
        # ----------- 自动维度修正 -----------
        if qv.ndim == 1:
            qv = qv.unsqueeze(0)
        if kv.ndim == 1:
            kv = kv.unsqueeze(0)

        # ----------- 安全检查 -----------
        if qv.ndim != 2 or kv.ndim != 2:
            raise ValueError(f"❌ Expect qv/kv 2D, got {qv.shape}, {kv.shape}")
        if qv.size(1) != self.d_proj or kv.size(1) != self.d_proj:
            raise ValueError(
                f"❌ Shape mismatch: qv={qv.shape}, kv={kv.shape}, expected (*, {self.d_proj})"
            )

        # ----------- 前向计算 -----------
        qn = F.normalize(self.q(qv), dim=-1)   # (B, d_proj)
        kn = F.normalize(self.k(kv), dim=-1)   # (B, d_proj)
        logits = qn @ kn.t() / self.temp.clamp_min(1e-3)
        labels = torch.arange(qn.size(0), device=qn.device)

        # ----------- 稳定交叉熵 -----------
        loss = F.cross_entropy(logits, labels)
        if torch.isnan(loss):
            print("⚠️ Warning: NaN loss detected, using zero fallback.")
            loss = torch.tensor(0.0, device=qn.device)

        return loss, logits