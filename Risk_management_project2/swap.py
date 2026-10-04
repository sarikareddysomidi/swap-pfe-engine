import numpy as np


class PayerSwap:
    def __init__(self, notional, fixed_rate, maturity, freq=1):
        self.notional = notional
        self.k = fixed_rate
        self.maturity = maturity
        self.freq = freq
        self.tau = 1.0 / freq
        self.pay_times = np.arange(1, int(maturity * freq) + 1) / freq

    def value(self, model, rates, times, i):
        t = times[i]
        remaining = self.pay_times[self.pay_times > t + 1e-9] - t
        if len(remaining) == 0:
            return np.zeros(rates.shape[1])
        past = self.pay_times[self.pay_times <= t + 1e-9]
        t_reset = past.max() if len(past) else 0.0
        j = int(round(t_reset / (times[1] - times[0])))
        fwd = (1.0 / model.bond_price(rates[j], self.tau) - 1.0) / self.tau
        p = np.stack([model.bond_price(rates[i], m) for m in remaining])
        float_leg = p[0] * (1.0 + fwd * self.tau) - p[-1]
        fixed_leg = self.k * self.tau * p.sum(axis=0)
        return self.notional * (float_leg - fixed_leg)

    def par_rate(self, model, r0):
        p = np.array([model.bond_price(np.array([r0]), m)[0] for m in self.pay_times])
        return (1.0 - p[-1]) / (self.tau * p.sum())
