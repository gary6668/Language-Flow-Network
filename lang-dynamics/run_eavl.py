import numpy as np, matplotlib.pyplot as plt, argparse, os
from scipy.stats import ttest_rel

def eval_plot(in_dir, out_dir):
    orig = np.loadtxt(os.path.join(in_dir, "mi_original.csv"))
    shuf = np.loadtxt(os.path.join(in_dir, "mi_shuffle.csv"))
    diff = orig - shuf
    t, p = ttest_rel(orig, shuf)
    print(f"Δ={diff.mean():.4f}, t={t:.2f}, p={p:.3e}")

    plt.figure()
    plt.bar(["Original", "Shuffled"], [orig.mean(), shuf.mean()], color=["blue", "orange"])
    plt.title(f"InfoNCE Comparison\nΔ={diff.mean():.3f}, p={p:.3e}")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "mi_compare.png"))
    plt.close()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", type=str)
    p.add_argument("--out_dir", type=str)
    args = p.parse_args()
    eval_plot(args.in_dir, args.out_dir)
