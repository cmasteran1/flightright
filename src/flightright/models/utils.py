# src/flightright/models/utils.py

import numpy as np

def enforce_monotone_ge_probs(p_ge: np.ndarray) -> np.ndarray:
    p = p_ge.copy()
    for j in range(1, p.shape[1]):
        p[:, j] = np.minimum(p[:, j], p[:, j - 1])
    return p

def ge_to_bins(p_ge15, p_ge30, p_ge45, p_ge60):
    p_ge = enforce_monotone_ge_probs(
        np.vstack([p_ge15, p_ge30, p_ge45, p_ge60]).T
    )
    p15, p30, p45, p60 = p_ge[:, 0], p_ge[:, 1], p_ge[:, 2], p_ge[:, 3]

    p_lt15  = 1.0 - p15
    p_15_30 = np.maximum(0.0, p15 - p30)
    p_30_45 = np.maximum(0.0, p30 - p45)
    p_45_60 = np.maximum(0.0, p45 - p60)
    p_ge60  = np.maximum(0.0, p60)

    P = np.vstack([p_lt15, p_15_30, p_30_45, p_45_60, p_ge60]).T
    Z = P.sum(axis=1, keepdims=True)
    Z[Z == 0] = 1.0
    return P / Z