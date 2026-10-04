import numpy as np


class Vasicek:
    def __init__(self, a=0.15, b=0.065, sigma=0.012, r0=0.06):
        self.a, self.b, self.sigma, self.r0 = a, b, sigma, r0

    def simulate(self, n_paths, n_steps, dt, seed=11):
        rng = np.random.default_rng(seed)
        a, b, s = self.a, self.b, self.sigma
        decay = np.exp(-a * dt)
        sd = s * np.sqrt((1 - np.exp(-2 * a * dt)) / (2 * a))
        r = np.empty((n_steps + 1, n_paths))
        r[0] = self.r0
        z = rng.standard_normal((n_steps, n_paths))
        for t in range(n_steps):
            r[t + 1] = r[t] * decay + b * (1 - decay) + sd * z[t]
        return r

    def bond_price(self, r, tau):
        a, b, s = self.a, self.b, self.sigma
        tau = np.asarray(tau, dtype=float)
        B = (1 - np.exp(-a * tau)) / a
        A = np.exp((b - s ** 2 / (2 * a ** 2)) * (B - tau) - s ** 2 * B ** 2 / (4 * a))
        return A * np.exp(-B * r)
