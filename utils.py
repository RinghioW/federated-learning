import numpy as np
from scipy.stats import entropy

def JS_div(p, q):
    M = (p + q) / 2
    j_s_div = 0.5 * entropy(p, M, base=2) + 0.5 * entropy(q, M, base=2)
    return j_s_div