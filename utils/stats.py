import numpy as np
from scipy.stats import ttest_rel

def paired_t(x, y):
    t, p = ttest_rel(x, y)
    d = (x.mean() - y.mean()) / np.std(x - y)
    return {"t": t, "p": p, "d": d}
