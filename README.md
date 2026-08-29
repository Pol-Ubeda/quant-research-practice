# Quant Research Practice

A collection of option pricing, calibration, and risk-management tools built from
scratch in Python — closed-form and lattice pricers, a full Heston stochastic-vol
implementation (calibration + Monte Carlo + semi-analytic pricing), a VaR/CVaR
portfolio risk engine, and dynamic hedging simulations. Everything pulls live
market data via `yfinance` rather than using synthetic inputs.

## Scripts

### Pricing models
| File | What it does |
|---|---|
| [`binomial_trees.py`](binomial_trees.py) | CRR binomial lattice pricer (European/American, calls/puts), with early-exercise boundary tracking and a convergence-to-BSM demo. |
| [`bsm_pricing_playground.py`](bsm_pricing_playground.py) | Closed-form Black-Scholes pricer with an animated price-vs-spot-and-time-to-expiry visualization. |
| [`heston.py`](heston.py) | Full Heston stochastic-volatility model: semi-analytic pricing via characteristic-function inversion, calibration to a live option chain (differential evolution + Nelder-Mead, vega-weighted loss, Feller-condition penalty), and a complete finite-difference Greeks suite. |
| [`autocallables.py`](autocallables.py) | Heston Monte Carlo pricer for autocallable structured notes (coupon barrier, autocall barrier, continuously-monitored knock-in put), plus finite-difference Greeks including the cross-Greeks (vanna, volga). |
| [`barrier_options.py`](barrier_options.py) | Heston Monte Carlo pricer for knock-in/knock-out barrier options, same Greeks suite as above. |
| [`mc_sims.py`](mc_sims.py) | GBM Monte Carlo option pricer, with simulated paths plotted against the realized price path for comparison. |

### Volatility
| File | What it does |
|---|---|
| [`iv_fitting.py`](iv_fitting.py) | Fetches a live option chain, solves implied vol per quote via Newton-Raphson, and returns an interpolated (strike, maturity) → IV surface. |
| [`garch.py`](garch.py) | GARCH(1,1) parameter estimation via MLE grid search, plus closed-form multi-day variance forecasting. |

### Risk & hedging
| File | What it does |
|---|---|
| [`VaR_engine.py`](VaR_engine.py) | Portfolio risk engine: builds a multi-option portfolio, computes Greeks (analytic BSM or binomial-lattice), and estimates VaR/CVaR via delta-normal, delta-gamma Monte Carlo, and historical simulation. Includes a greek-attribution P&L-explain report. |
| [`Greeks.py`](Greeks.py) | BSM Greeks plus dynamic delta/gamma/vega hedging simulations — multi-instrument hedge ratios solved via linear algebra, verified against a Monte Carlo P&L distribution. |

## Setup

```bash
pip install -r requirements.txt
```

Each script is runnable standalone:

```bash
python heston.py
```
