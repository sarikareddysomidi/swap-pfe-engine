import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.vasicek import Vasicek
from src.swap import PayerSwap
from src.collateral import apply_vm
from src.exposure import exposure_profile, effective_epe, cva

N_PATHS = 4000
MATURITY = 5
STEPS_PER_YEAR = 252
MPOR = 10
NOTIONAL = 100_000_000


def main():
    os.makedirs("output", exist_ok=True)
    model = Vasicek()
    dt = 1.0 / STEPS_PER_YEAR
    n_steps = MATURITY * STEPS_PER_YEAR
    rates = model.simulate(N_PATHS, n_steps, dt)
    times = np.arange(n_steps + 1) * dt

    swap = PayerSwap(NOTIONAL, 0.0, MATURITY)
    swap.k = swap.par_rate(model, model.r0) + 0.002

    values = np.stack([swap.value(model, rates, times, i) for i in range(n_steps + 1)])
    disc = np.exp(-np.cumsum(rates, axis=0) * dt).mean(axis=1)

    setups = {
        "uncollateralised": np.zeros_like(values),
        "VM zero threshold": apply_vm(values, MPOR, 0.0, 0.0),
        "VM threshold 5m, MTA 0.5m": apply_vm(values, MPOR, 5e6, 5e5),
    }

    rows = []
    profiles = {}
    for name, coll in setups.items():
        prof = exposure_profile(values, coll, times)
        profiles[name] = prof
        rows.append({
            "setup": name,
            "peak_PFE_95": round(prof["PFE"].max()),
            "EEPE_1y": round(effective_epe(prof["EE"].values, times)),
            "CVA": round(cva(prof["EE"].values, times, disc)),
        })

    print(f"par + 20bp fixed rate: {swap.k:.4%}\n")
    for r in rows:
        print(r)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, prof in profiles.items():
        ax[0].plot(prof["time"], prof["EE"] / 1e6, label=name)
        ax[1].plot(prof["time"], prof["PFE"] / 1e6, label=name)
    ax[0].set_title("Expected exposure (m)")
    ax[1].set_title("95% PFE (m)")
    ax[0].legend()
    for a in ax:
        a.set_xlabel("years")
    plt.tight_layout()
    plt.savefig("output/exposure_profiles.png", dpi=130)

    out = profiles["uncollateralised"][["time"]].copy()
    for name, prof in profiles.items():
        out[name + " EE"] = prof["EE"].round(0)
        out[name + " PFE"] = prof["PFE"].round(0)
    out.iloc[::21].to_csv("output/exposure_profiles.csv", index=False)


if __name__ == "__main__":
    main()
