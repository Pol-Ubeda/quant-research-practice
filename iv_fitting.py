import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import datetime as dt
from scipy.stats import norm
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from mpl_toolkits.mplot3d import Axes3D


def fetch_all_chains(ticker, min_dte=30, max_dte=540):
    """
    Download option chains for all expiries within [min_dte, max_dte] days.
    Returns spot price S and a list of (T, expiry_str, calls_df, puts_df).
    """
    stock  = yf.Ticker(ticker)
    S      = stock.fast_info["last_price"]
    today  = dt.date.today()
    chains = []

    for expiry_str in stock.options:
        expiry_date = dt.date.fromisoformat(expiry_str)
        dte = (expiry_date - today).days
        if not (min_dte <= dte <= max_dte):
            continue
        T     = dte / 252
        chain = stock.option_chain(expiry_str)
        chains.append((T, expiry_str, chain.calls, chain.puts))
        print(f"Fetched {expiry_str}  (T={T:.3f}yr)")

    return S, chains


def bsm_price(S0, K, r, T, vol, option_type="call"):
    d1 = (np.log(S0/K) + (r + 0.5*vol**2)*T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    if option_type == "call":
        return S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2) - S0*norm.cdf(-d1)


def compute_vega(S, K, r, vol, T):
    d1 = (np.log(S/K) + (r + 0.5*vol**2)*T) / (vol * np.sqrt(T))
    return S * np.sqrt(T) * norm.pdf(d1)


def newton_raphson(S, K, r, T, market_price, option_type, vol0=0.3, max_iter=100, tol=1e-4):
    vol = vol0
    for _ in range(max_iter):
        price = bsm_price(S, K, r, T, vol, option_type)
        err   = price - market_price
        if abs(err) < tol:
            break
        vega = compute_vega(S, K, r, vol, T)
        if vega < 1e-10:
            return None
        vol -= err / vega
        if vol <= 0 or vol > 10:
            return None
    return vol if 0 < vol < 10 else None


def build_vol_surface(S, chains, r, option_type="call"):
    """
    For each expiry, compute IVs across strikes.
    Returns arrays: strikes, maturities, ivs — one entry per valid (K, T) pair.
    """
    strikes, maturities, ivs = [], [], []

    for T, expiry_str, calls, puts in chains:
        contracts = calls if option_type == "call" else puts
        for _, row in contracts.iterrows():
            K      = row["strike"]
            spread = row["ask"] - row["bid"]
            mid    = (row["ask"] + row["bid"]) / 2
            if mid <= 0 or (spread / mid) > 0.5:
                continue
            iv = newton_raphson(S, K, r, T, mid, option_type)
            if iv is None:
                continue
            strikes.append(K)
            maturities.append(T)
            ivs.append(iv)

    return np.array(strikes), np.array(maturities), np.array(ivs)


def build_iv_interpolator(ticker, r, option_type="call"):
    """
    Fetch the IV surface for a ticker and return a callable interpolator.

    The interpolator takes (strike, maturity) and returns implied vol.
    Inside the surface it uses linear interpolation; outside it falls back
    to the nearest observed point (no extrapolation artefacts).

    Usage:
        iv_lookup = build_iv_interpolator("AAPL", r=0.05)
        vol = iv_lookup(150.0, 0.5)   # strike=150, T=0.5yr
    """
    S, chains           = fetch_all_chains(ticker)
    strikes, mats, ivs  = build_vol_surface(S, chains, r, option_type)

    points = np.column_stack([strikes, mats])   # shape (n, 2)

    # Primary: linear interpolation inside the convex hull of observed points
    linear  = LinearNDInterpolator(points, ivs)
    # Fallback: nearest-neighbour for queries outside the hull
    nearest = NearestNDInterpolator(points, ivs)

    def lookup(strike, maturity):
        pt  = np.array([[strike, maturity]])
        val = linear(pt)[0]
        if np.isnan(val):               # outside the surface — use nearest
            val = nearest(pt)[0]
        return float(val)

    return lookup


def plot_vol_surface(strikes, maturities, ivs, ticker, option_type):
    fig = plt.figure(figsize=(12, 7))
    ax  = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(strikes, maturities, ivs, cmap='viridis', alpha=0.8)
    ax.set_xlabel('Strike')
    ax.set_ylabel('Maturity (yrs)')
    ax.set_zlabel('Implied Volatility')
    ax.set_title(f'{ticker} {option_type.capitalize()} Implied Volatility Surface')
    plt.tight_layout()
    plt.show()


def main():
    ticker = 'TSLA'
    r      = 0.038

    S, chains = fetch_all_chains(ticker, min_dte=30, max_dte=540)
    print(f"\nSpot: {S:.2f} | {len(chains)} expiries loaded\n")

    call_strikes, call_mats, call_ivs = build_vol_surface(S, chains, r, "call")
    put_strikes,  put_mats,  put_ivs  = build_vol_surface(S, chains, r, "put")

    plot_vol_surface(call_strikes, call_mats, call_ivs, ticker, "call")
    plot_vol_surface(put_strikes,  put_mats,  put_ivs,  ticker, "put")


if __name__ == '__main__':
    main()
