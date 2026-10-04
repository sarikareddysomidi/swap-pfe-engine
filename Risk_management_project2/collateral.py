import numpy as np


def apply_vm(values, mpor_steps, threshold=0.0, mta=0.0):
    n_steps, n_paths = values.shape
    coll = np.zeros_like(values)
    held = np.zeros(n_paths)
    for t in range(n_steps):
        lag = max(t - mpor_steps, 0)
        v = values[lag]
        required = np.where(v > threshold, v - threshold,
                            np.where(v < -threshold, v + threshold, 0.0))
        move = required - held
        call = np.abs(move) >= mta
        held = np.where(call, required, held)
        coll[t] = held
    return coll
