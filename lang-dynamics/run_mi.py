import numpy as np, torch, argparse, os
from tqdm import tqdm
from models.mine import WindowEncoder, InfoNCE

def window_samples(arr, m=3):
    out = []
    for i in range(m, arr.shape[0]):
        out.append((arr[i - m:i], arr[i]))
    return out

def run_one(npz, window=3, mode="original", out_path=None):
    data = np.load(npz, allow_pickle=True)['emb']
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mi_est = InfoNCE().to(device)
    encoder = WindowEncoder(768, window).to(device)
    all_losses = []

    for doc in tqdm(data):
        if doc.shape[0] <= window + 1: continue
        if mode == "shuffle":
            np.random.shuffle(doc)
        pairs = window_samples(doc, m=window)
        windows = torch.tensor([x[0] for x in pairs], dtype=torch.float32).to(device)
        targets = torch.tensor([x[1] for x in pairs], dtype=torch.float32).to(device)
        with torch.no_grad():
            qv = encoder(windows)
            kv = targets.mean(dim=1)
            loss, _ = mi_est(qv, kv)
        all_losses.append(loss.item())

    res = np.mean(all_losses)
    print(f"{mode} mean InfoNCE loss: {res:.4f}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savetxt(out_path, [res])
    return res

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--emb", type=str)
    p.add_argument("--window", type=int, default=3)
    p.add_argument("--mode", type=str, default="original")
    p.add_argument("--out", type=str)
    args = p.parse_args()
    run_one(args.emb, args.window, args.mode, args.out)
