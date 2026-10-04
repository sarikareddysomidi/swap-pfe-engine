# Swap Exposure and PFE Engine

Monte Carlo counterparty exposure for a 5-year interest rate swap under a Vasicek short-rate model, comparing uncollateralised exposure with variation margin under different CSA terms.

## Components

- `src/vasicek.py` exact-discretisation rate simulation and closed-form zero-coupon bond prices
- `src/swap.py` payer swap valuation with accrued floating coupon
- `src/collateral.py` variation margin with threshold, minimum transfer amount and margin period of risk
- `src/exposure.py` expected exposure, PFE, effective EPE, CVA
- `run.py` runs three CSA setups and writes `output/`

## Run

    pip install -r requirements.txt
    python run.py

## Setups compared

- uncollateralised
- daily VM, zero threshold, 10-day margin period of risk
- daily VM, 5m threshold, 0.5m minimum transfer amount

## Assumptions

- single-curve, single-factor model with constant parameters
- flat hazard rate and fixed recovery for CVA
- no initial margin, no funding adjustments

## Next steps

- calibrate Vasicek parameters to market data
- add FX forwards and a second risk factor
- add initial margin and MVA
