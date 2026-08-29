import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
from dataclasses import dataclass, field
from datetime import date
from iv_fitting import build_iv_interpolator
from iv_fitting import bsm_price
from binomial_trees import build_lattice

@dataclass
class Option:
    ticker: str
    strike: float
    expiry: date
    spot: float
    option_style: str     # "american" or "european"
    option_type: str      # "call" or "put"
    position: float       # number of contracts (negative = short)
    delta: float
    gamma: float
    theta: float
    vega: float
    volatility: float = 0.0

    @property
    def maturity(self) -> float:
        return (self.expiry - date.today()).days / 252.0

    def compute_greeks(self, spot: float, risk_free_rate: float) -> None:
        if(self.option_style == "european"):
            S, K = spot, self.strike
            T, r = self.maturity, risk_free_rate
            vol = self.volatility

            d1 = (np.log(S / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
            d2 = d1 - (vol * np.sqrt(T))

            if self.option_type == "call":
                self.delta = norm.cdf(d1)
                self.theta = (-(S * norm.pdf(d1) * vol) / (2 * np.sqrt(T))
                              - r * K * np.exp(-r * T) * norm.cdf(d2)) / 252
            else:
                self.delta = norm.cdf(d1) - 1
                self.theta = (-(S * norm.pdf(d1) * vol) / (2 * np.sqrt(T))
                              + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 252

            self.gamma = norm.pdf(d1) / (S * vol * np.sqrt(T))
            self.vega  = S * np.sqrt(T) * norm.pdf(d1) / 100  # per 1% move in vol
        else:
            S_eps   = 0.01 * self.spot   # spot bump for delta/gamma
            vol_eps = 0.001              # vol bump for vega (0.1% vol)
            n       = 1000
            dt      = self.maturity / n  # maturity in years, not a date object

            # base u, d at current vol — reused for delta, gamma, theta
            u = np.exp( self.volatility * np.sqrt(dt))
            d = np.exp(-self.volatility * np.sqrt(dt))

            def lattice_price(S0, u, d, dt):
                result = build_lattice(n, u, d, S0, dt, self.strike,
                                       self.option_style, self.option_type,
                                       risk_free_rate)
                lattice = result[0] if isinstance(result, tuple) else result
                return lattice[0][0].option_value

            # delta and gamma: bump spot, keep vol fixed
            price_topS = lattice_price(self.spot + S_eps, u, d, dt)
            price_midS = lattice_price(self.spot,         u, d, dt)
            price_lowS = lattice_price(self.spot - S_eps, u, d, dt)

            self.delta = (price_topS - price_lowS) / (2 * S_eps)
            self.gamma = (price_topS - 2 * price_midS + price_lowS) / (S_eps ** 2)

            # vega: bump vol, keep spot fixed — vol enters through u and d
            u_hi = np.exp( (self.volatility + vol_eps) * np.sqrt(dt))
            d_hi = np.exp(-(self.volatility + vol_eps) * np.sqrt(dt))
            u_lo = np.exp( (self.volatility - vol_eps) * np.sqrt(dt))
            d_lo = np.exp(-(self.volatility - vol_eps) * np.sqrt(dt))

            price_topV = lattice_price(self.spot, u_hi, d_hi, dt)
            price_lowV = lattice_price(self.spot, u_lo, d_lo, dt)
            self.vega  = (price_topV - price_lowV) / (2 * vol_eps) / 100  # per 1% vol move

            # theta: bump time by one day, keep vol fixed (use base u, d)
            dt_shifted = (self.maturity - 1/252) / n
            price_lowT = lattice_price(self.spot, u, d, dt_shifted)
            self.theta = price_lowT - price_midS   # reuse price_midS (same S, same vol)

class Portfolio:

    def __init__(self):
        self._options: list[Option] = []
        self._returns: dict[str, pd.Series] = {}

    def add_option(self, ticker, strike, expiry, spot, option_style, option_type, position, delta=0.0, gamma=0.0, theta=0.0, vega=0.0):
        opt = Option(ticker, strike, expiry, spot, option_style,  option_type, position, delta, gamma, theta, vega)
        self._options.append(opt)
        if ticker not in self._returns:
            self._returns[ticker] = download_log_returns(ticker)

    def _returns_matrix(self) -> pd.DataFrame:
        return pd.DataFrame(self._returns).dropna()

    def compute_covariance(self) -> pd.DataFrame:
        return self._returns_matrix().cov()

    def compute_correlations(self) -> pd.DataFrame:
        cov = self.compute_covariance().values
        vols = np.sqrt(np.diag(cov))
        outer_vols = np.outer(vols, vols)
        corr_values = cov / outer_vols
        tickers = list(self._returns.keys())
        return pd.DataFrame(corr_values, index=tickers, columns=tickers)

    def dollar_delta_vol(self) -> float:
        """Daily portfolio std dev: sqrt(alpha^T Sigma alpha), alpha = dollar-delta per ticker."""
        cov = self.compute_covariance()
        tickers = list(self._returns.keys())
        alpha = {}
        for opt in self._options:
            alpha[opt.ticker] = alpha.get(opt.ticker, 0.0) + opt.delta * opt.spot * opt.position
        alpha_vec = np.array([alpha.get(t, 0.0) for t in tickers])
        return float(np.sqrt(alpha_vec @ cov.values @ alpha_vec))

    def fit_volatilities(self, vol_fn):
        """vol_fn receives the full Option object so it can use strike and maturity."""
        for opt in self._options:
            opt.volatility = vol_fn(opt)

    def summary(self):
        print(f"{'Ticker':<8} {'Type':<5} {'Strike':>8} {'Maturity':>10} "
              f"{'Vol':>8} {'Delta':>8} {'Gamma':>8} {'Position':>10}")
        print("-" * 70)
        for opt in self._options:
            print(f"{opt.ticker:<8} {opt.option_type:<5} {opt.strike:>8.2f} "
                  f"{opt.maturity:>10.4f} {opt.volatility:>8.4f} "
                  f"{opt.delta:>8.4f} {opt.gamma:>8.4f} {opt.position:>10}")

    def __len__(self):
        return len(self._options)

    def __iter__(self):
        return iter(self._options)


def download_ohlcv(ticker, time_span, interval):
    df = yf.download(
        ticker,
        period=time_span,
        interval=interval,
        auto_adjust=False,
        multi_level_index=False
    )
    # yfinance sometimes leaves Close NaN on the most recent (unsettled) row
    return df.dropna(subset=["Close"])

def download_log_returns(ticker: str, period: str = "1y") -> pd.Series:
    raw = yf.download(ticker, period=period, auto_adjust=False,
                      multi_level_index=False)["Close"]
    return np.log(raw / raw.shift(1)).dropna()

def annualised_vol(df, col="Close"):
    prices = df[col]
    log_returns = np.log(prices / prices.shift(1)).dropna()
    return log_returns.std(ddof=1) * np.sqrt(252)

def historical_simulation_var(df, percentile):
    closing_prices = df["Close"]
    daily_returns = closing_prices.pct_change().dropna()
    var_return  = daily_returns.quantile(percentile)
    cvar_return = daily_returns[daily_returns < var_return].mean()
    S = closing_prices.iloc[-1]
    var_dollar  = float(-S * var_return)
    cvar_dollar = float(-S * cvar_return)

    plt.hist(daily_returns, bins=50, density=True, alpha=0.7, edgecolor='black')
    plt.axvline(var_return,  color='red',    linestyle='dashed', linewidth=2,
                label=f'{int((1-percentile)*100)}% VaR ({var_return:.2%})')
    plt.axvline(cvar_return, color='orange', linestyle='dashed', linewidth=2,
                label=f'{int((1-percentile)*100)}% CVaR ({cvar_return:.2%})')
    plt.xlabel('Daily Return')
    plt.ylabel('Density')
    plt.title('Historical Returns Distribution (Historical Simulation VaR)')
    plt.legend()
    plt.show()
    return var_dollar, cvar_dollar

def delta_normal_var(percentile, vol, S):
    """Single-asset parametric VaR and CVaR."""
    daily_vol = vol / np.sqrt(252) * S
    z = norm.ppf(percentile)
    var  = -z * daily_vol
    cvar = norm.pdf(z) / percentile * daily_vol
    return var, cvar

def delta_normal_var_portfolio(portfolio: Portfolio, percentile):
    """Multi-asset parametric VaR using delta-normal approximation."""
    return -norm.ppf(percentile) * portfolio.dollar_delta_vol()

def delta_gamma_mc_var(portfolio: Portfolio, percentile, n_sims=100_000, seed=42, plot=True):
    """
    Monte Carlo VaR using a delta-gamma (quadratic) P&L approximation.
    Simulates correlated returns via Cholesky, applies dP = alpha·r + beta·r²,
    and reads off the loss percentile.
    """
    np.random.seed(seed)

    cov = portfolio.compute_covariance()
    tickers = list(cov.columns)
    cov_matrix = cov.values

    alpha = np.zeros(len(tickers))
    beta  = np.zeros(len(tickers))
    for opt in portfolio:
        i = tickers.index(opt.ticker)
        alpha[i] += opt.delta * opt.spot * opt.position
        beta[i]  += 0.5 * opt.gamma * (opt.spot ** 2) * opt.position

    L = np.linalg.cholesky(cov_matrix)
    z_samples = np.random.standard_normal((n_sims, len(tickers)))
    r_samples = z_samples @ L.T

    pnl = r_samples @ alpha + (r_samples ** 2) @ beta
    var  = float(-np.percentile(pnl, percentile * 100))
    cvar = float(-pnl[pnl < -var].mean())

    if plot:
        plt.hist(pnl, bins=100, density=True, alpha=0.7, edgecolor='black')
        plt.axvline(-var,  color='red',    linestyle='dashed', linewidth=2,
                    label=f'{int((1-percentile)*100)}% VaR (${var:.2f})')
        plt.axvline(-cvar, color='orange', linestyle='dashed', linewidth=2,
                    label=f'{int((1-percentile)*100)}% CVaR (${cvar:.2f})')
        plt.xlabel('Portfolio P&L ($)')
        plt.ylabel('Density')
        plt.title('Delta-Gamma MC — Simulated P&L Distribution')
        plt.legend()
        plt.show()

    return var, cvar, pnl


def show_option_chain(ticker, expiry=None):
    """
    Browse available options for a ticker.
    - No expiry given: prints all available expiry dates.
    - Expiry given (e.g. "2026-09-18"): prints strikes and mid prices for that date.
    """
    stock = yf.Ticker(ticker)
    S     = stock.fast_info["last_price"]

    if expiry is None:
        print(f"\n{ticker} — current spot: ${S:.2f}")
        print(f"{'#':<4} {'Expiry':<12} {'DTE':>5}")
        print("-" * 24)
        today = date.today()
        for i, e in enumerate(stock.options):
            dte = (date.fromisoformat(e) - today).days
            print(f"{i:<4} {e:<12} {dte:>5}d")
    else:
        chain  = stock.option_chain(expiry)
        today  = date.today()
        T      = (date.fromisoformat(expiry) - today).days / 252.0
        print(f"\n{ticker} | expiry={expiry} | T={T:.3f}yr | spot=${S:.2f}")
        for label, contracts in [("CALLS", chain.calls), ("PUTS", chain.puts)]:
            print(f"\n  {label}")
            print(f"  {'Strike':>8} {'Bid':>7} {'Ask':>7} {'Mid':>7} {'IV':>7} {'Volume':>8}")
            print(f"  {'-'*50}")
            for _, row in contracts.iterrows():
                mid = (row["bid"] + row["ask"]) / 2
                iv  = row.get("impliedVolatility", float("nan"))
                print(f"  {row['strike']:>8.1f} {row['bid']:>7.2f} {row['ask']:>7.2f} "
                      f"{mid:>7.2f} {iv:>7.2%} {int(row['volume'] or 0):>8}")


def reprice_option(S, opt, r, T, vol, n=1000):
    """Reprice at a given spot/maturity/vol — BSM for European, a fresh
    binomial lattice for American (same pricing method compute_greeks uses)."""
    if opt.option_style == "european":
        return bsm_price(S, opt.strike, r, T, vol, opt.option_type)
    dt_step = T / n
    u = np.exp( vol * np.sqrt(dt_step))
    d = np.exp(-vol * np.sqrt(dt_step))
    result = build_lattice(n, u, d, S, dt_step, opt.strike,
                           opt.option_style, opt.option_type, r)
    lattice = result[0] if isinstance(result, tuple) else result
    return lattice[0][0].option_value


def pnl_explain(portfolio, r, iv_interpolators=None):
    """
    Decompose yesterday→today P&L for each option into greek contributions.

    For each option:
      - Fetches yesterday's and today's closing spot price
      - Looks up today's implied vol at (strike, new maturity) for dSigma
      - Computes: delta, gamma, vega, theta contributions and residual
      - Residual = actual repriced P&L minus the sum of greek contributions
    """
    today     = date.today()
    print(f"\n{'='*75}")
    print(f"  P&L Explain — {today}")
    print(f"{'='*75}")
    print(f"{'Option':<14} {'dS':>7} {'dSigma%':>7} {'Delta':>8} {'Gamma':>8} "
          f"{'Vega':>8} {'Theta':>8} {'Expl.':>8} {'Actual':>8} {'Resid.':>8}")
    print(f"{'-'*75}")

    rows  = []
    total = {k: 0.0 for k in ["delta", "gamma", "vega", "theta", "explained", "actual", "residual"]}

    for opt in portfolio:
        raw     = yf.download(opt.ticker, period="5d", auto_adjust=False,
                              multi_level_index=False)["Close"].dropna()
        S_prev  = float(raw.iloc[-2])
        S_today = float(raw.iloc[-1])
        dS      = S_today - S_prev

        T_today = (opt.expiry - today).days / 252.0
        if iv_interpolators and iv_interpolators.get(opt.ticker):
            sigma_today = iv_interpolators[opt.ticker](opt.strike, T_today)
        else:
            sigma_today = opt.volatility

        # No day-over-day IV history is persisted, so yesterday's IV is
        # assumed equal to today's (dSigma = 0) unless the interpolator itself
        # has moved between calls.
        sigma_prev = opt.volatility
        d_sigma = sigma_today - sigma_prev

        delta_pnl = opt.delta * dS           * opt.position
        gamma_pnl = 0.5 * opt.gamma * dS**2  * opt.position
        vega_pnl  = opt.vega * (d_sigma*100)  * opt.position
        theta_pnl = opt.theta                 * opt.position
        explained = delta_pnl + gamma_pnl + vega_pnl + theta_pnl

        # NOTE: actual P&L reprices via the model (BSM for European, binomial
        # lattice for American) as a proxy — replace with real market option
        # prices once a historical options data source is available.
        price_prev  = reprice_option(S_prev,  opt, r, opt.maturity + 1/252,
                                     opt.volatility) * opt.position
        price_today = reprice_option(S_today, opt, r, T_today,
                                     sigma_today)    * opt.position
        actual   = price_today - price_prev
        residual = actual - explained

        row = {"Option": f"{opt.ticker} {opt.option_type}",
               "dS": dS, "dSigma%": d_sigma*100,
               "Delta P&L": delta_pnl, "Gamma P&L": gamma_pnl,
               "Vega P&L": vega_pnl,  "Theta P&L": theta_pnl,
               "Explained": explained, "Actual": actual, "Residual": residual}
        rows.append(row)

        label = f"{opt.ticker} {opt.option_type}"
        print(f"{label:<14} {dS:>+7.2f} {d_sigma*100:>+7.2f} {delta_pnl:>+8.2f} "
              f"{gamma_pnl:>+8.2f} {vega_pnl:>+8.2f} {theta_pnl:>+8.2f} "
              f"{explained:>+8.2f} {actual:>+8.2f} {residual:>+8.2f}")

        for k, v in zip(total, [delta_pnl, gamma_pnl, vega_pnl, theta_pnl,
                                 explained, actual, residual]):
            total[k] += v

    print(f"{'-'*75}")
    print(f"{'TOTAL':<14} {'':>7} {'':>7} {total['delta']:>+8.2f} "
          f"{total['gamma']:>+8.2f} {total['vega']:>+8.2f} {total['theta']:>+8.2f} "
          f"{total['explained']:>+8.2f} {total['actual']:>+8.2f} {total['residual']:>+8.2f}")
    print(f"{'='*75}\n")

    df = pd.DataFrame(rows)
    total_row = pd.DataFrame([{"Option": "TOTAL", "dS": "", "dSigma%": "",
                                **{k: v for k, v in zip(
                                    ["Delta P&L","Gamma P&L","Vega P&L","Theta P&L",
                                     "Explained","Actual","Residual"], total.values())}}])
    return pd.concat([df, total_row], ignore_index=True)


def main():
    ticker = "AAPL"
    percentile = 0.01

    # Historical simulation VaR
    df = download_ohlcv(ticker, "2y", "1d")
    var, cvar = historical_simulation_var(df, percentile)
    print(f"Historical simulation  — VaR: ${var:.2f}  CVaR: ${cvar:.2f}")

    # Single-asset delta-normal VaR
    S = float(df["Close"].iloc[-1])
    var, cvar = delta_normal_var(percentile, annualised_vol(df), S)
    print(f"Delta-normal (single)  — VaR: ${var:.2f}  CVaR: ${cvar:.2f}")


    r = 0.05
    selections = [
        {"ticker": "AAPL", "expiry": "2026-09-18", "strike": 230.0,
         "option_type": "call", "option_style": "american", "position": 1},
        {"ticker": "MSFT", "expiry": "2026-09-18", "strike": 380.0,
         "option_type": "put",  "option_style": "american", "position": 1},
    ]

    # build the portfolio: fetch spot, add each option
    p = Portfolio()
    for s in selections:
        spot = yf.Ticker(s["ticker"]).fast_info["last_price"]
        p.add_option(
            ticker=s["ticker"], strike=s["strike"],
            expiry=date.fromisoformat(s["expiry"]), spot=spot,
            option_style=s["option_style"], option_type=s["option_type"],
            position=s["position"],
        )

    # fit vols from each ticker's IV surface, falling back to historical vol
    tickers = list({s["ticker"] for s in selections})
    iv_interpolators = {}
    for t in tickers:
        try:
            iv_interpolators[t] = build_iv_interpolator(t, r)
        except Exception as e:
            print(f"IV surface unavailable for {t} ({e}), falling back to historical vol")
            iv_interpolators[t] = None

    hist_dfs = {}
    def iv_lookup(opt):
        interp = iv_interpolators.get(opt.ticker)
        if interp is not None:
            return interp(opt.strike, opt.maturity)
        if opt.ticker not in hist_dfs:
            hist_dfs[opt.ticker] = download_ohlcv(opt.ticker, "1y", "1d")
        return annualised_vol(hist_dfs[opt.ticker])

    p.fit_volatilities(iv_lookup)
    for opt in p:
        opt.compute_greeks(opt.spot, r)

    p.summary()
    print(f"\nCorrelation matrix:\n{p.compute_correlations().round(4)}")

    var_linear                    = delta_normal_var_portfolio(p, percentile)
    var_quad, cvar_quad, _        = delta_gamma_mc_var(p, percentile)
    print(f"\nDelta-normal (portfolio) — VaR: ${var_linear:.2f}")
    print(f"Delta-gamma MC           — VaR: ${var_quad:.2f}  CVaR: ${cvar_quad:.2f}")

    pnl_explain(p, r, iv_interpolators)

if __name__ == "__main__":
    main()
