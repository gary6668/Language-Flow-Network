import numpy as np
import torch
import argparse
import os
from tqdm import tqdm
from models.mine import WindowEncoder, InfoNCE


def window_samples(arr, m=3):
    out = []
    for i in range(m, arr.shape[0]):
        out.append((arr[i - m:i], arr[i]))
    return out


def run_one(npz, window=3, mode="original", out_path=None):
    print(f"\n---- Running MI ({mode}) ----")

    data = np.load(npz, allow_pickle=True)["emb"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder = WindowEncoder(d=768, m=window, d_proj=256).to(device)
    mi_est = InfoNCE(d_proj=256).to(device)

    all_losses = []

    for doc in tqdm(data, desc=f"Docs ({mode})"):
        try:
            # --- 自动清理 object 数组 ---
            if isinstance(doc, np.ndarray) and doc.dtype == object:
                # 拉平成统一 float 数组
                doc = np.vstack([np.array(x, dtype=np.float32) for x in doc if isinstance(x, np.ndarray)])
            else:
                doc = np.array(doc, dtype=np.float32)

            # --- 跳过异常样本 ---
            if doc.ndim != 2 or doc.shape[0] <= window + 1:
                continue

            # --- 模式扰动 ---
            if mode == "shuffle_layer":
                np.random.shuffle(doc)
            elif mode == "shuffle_time":
                doc = doc[::-1].copy()

            # --- 构造窗口样本 ---
            pairs = window_samples(doc, m=window)
            windows = torch.tensor(np.stack([x[0] for x in pairs]), dtype=torch.float32).to(device)
            targets = torch.tensor(np.stack([x[1] for x in pairs]), dtype=torch.float32).to(device)

            with torch.no_grad():
                qv = encoder(windows)
                if targets.ndim == 2:
                    kv = targets[:, :qv.size(1)]
                elif targets.ndim == 3:
                    kv = targets.mean(dim=1)
                else:
                    kv = targets.unsqueeze(0)
                loss, _ = mi_est(qv, kv)

            if not torch.isnan(loss):
                all_losses.append(loss.item())

        except Exception as e:
            print(f"⚠️ Skipped doc due to error: {e}")
            continue

    res = float(np.mean(all_losses)) if len(all_losses) > 0 else float("nan")
    print(f"{mode} mean InfoNCE loss: {res:.4f}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savetxt(out_path, [res])
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--emb", type=str, required=True)
    p.add_argument("--window", type=int, default=3)
    p.add_argument("--mode", type=str, default="original",
                   choices=["original", "shuffle_time", "shuffle_layer"])
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    run_one(args.emb, args.window, args.mode, args.out)