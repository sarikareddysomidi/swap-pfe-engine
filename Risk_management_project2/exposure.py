import numpy as np
import pandas as pd


def exposure_profile(values, collateral, times, pfe_level=0.95):
    net = np.maximum(values - collateral, 0.0)
    return pd.DataFrame({
        "time": times,
        "EE": net.mean(axis=1),
        "PFE": np.quantile(net, pfe_level, axis=1),
    })


def effective_epe(ee, times, horizon=1.0):
    mask = times <= horizon
    running = np.maximum.accumulate(ee[mask])
    dt = np.diff(times[mask], prepend=0.0)
    return float((running * dt).sum() / max(times[mask][-1], 1e-9))


def cva(ee, times, discount, hazard=0.02, recovery=0.4):
    surv = np.exp(-hazard * times)
    pd_marg = -np.diff(surv, prepend=1.0)
    return float((1 - recovery) * (ee * discount * pd_marg).sum())
