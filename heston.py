import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Patch
from iv_fitting import fetch_all_chains, newton_raphson, compute_vega, bsm_price

def compute_CF(x0, r, rho, sigma, kappa, lra, v0, phi, tau, i):
    if(i == 1):
        u = 0.5
        b = kappa - rho*sigma
    else:
        u = -0.5
        b = kappa
    d = np.sqrt((rho*sigma*1j*phi - b)**2 - sigma**2*(2*u*1j*phi - phi**2))
    c = (b - rho*sigma*1j*phi - d) / (b - rho*sigma*1j*phi + d)

    #Following are paramters for the characteristic functions, and the characteristic functions f1, f2
    D = ((b - rho*sigma*1j*phi - d) /  sigma**2) * ((1 - np.exp(-d*tau)) / (1 - c*np.exp(-d*tau)))
    C = r*1j*phi*tau + kappa*lra/sigma**2 * ((b - rho*sigma*1j*phi - d)*tau - 2*np.log((1 - c*np.exp(-d*tau))/(1 - c)))
    f = np.exp(C + D*v0 +1j*phi*x0)
    return f

def heston_price(x0, r, K, rho, sigma, kappa, lra, v0, tau):
    phi = np.linspace(1e-8, 100, 200)
    
    integrand = np.imag(np.exp(-1j*phi*np.log(K)) * compute_CF(x0, r, rho, sigma, kappa, lra, v0, phi, tau, 1) / phi)
    P1 = 0.5 + np.trapezoid(integrand, phi) / np.pi

    integrand = np.imag(np.exp(-1j*phi*np.log(K)) * compute_CF(x0, r, rho, sigma, kappa, lra, v0, phi, tau, 2) / phi)
    P2 = 0.5 + np.trapezoid(integrand, phi) / np.pi
    
    option_price = np.exp(x0) * P1 - K * np.exp(-r * tau) * P2
    return option_price


def get_option_chain(ticker, r, min_dte=30, max_dte=540, option_type="call"):
    """
    Fetch a flat DataFrame of options for calibration.
    Columns: strike, T, expiry, bid, ask, mid
    Filters out illiquid contracts (zero mid or spread > 50% of mid).
    """
    S, chains = fetch_all_chains(ticker, min_dte=min_dte, max_dte=max_dte)
    rows = []
    for T, expiry_str, calls, puts in chains:
        contracts = calls if option_type == "call" else puts
        for _, row in contracts.iterrows():
            bid, ask = row["bid"], row["ask"]
            mid = (bid + ask) / 2
            if mid <= 0 or (ask - bid) / mid > 0.5:
                continue
            rows.append({"strike": row["strike"], "T": T, "expiry": expiry_str,
                         "bid": bid, "ask": ask, "mid": mid})
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} {option_type} contracts for {ticker}  (S={S:.2f})")
    return S, df


def compute_market_ivs(df, S, r, option_type="call"):
    """Add iv_market and bsm_price columns by inverting BSM on the mid price."""
    df = df.copy()
    df["iv_market"] = [newton_raphson(S, row.strike, r, row.T, row.mid, option_type)
                       for row in df.itertuples()]
    df.dropna(subset=["iv_market"], inplace=True)
    df["bsm_price"] = [bsm_price(S, row.strike, r, row.T, row.iv_market, option_type)
                       for row in df.itertuples()]
    return df.reset_index(drop=True)


def compute_heston_ivs(df, S, r, params, option_type="call"):
    """
    Price each option with the Heston model then back out the BSM IV.
    params = (kappa, lra, sigma, rho, v0)
    Adds iv_heston column; rows where pricing fails are dropped.
    """
    kappa, lra, sigma, rho, v0 = params
    x0 = np.log(S)
    heston_prices = []
    heston_ivs = []
    for row in df.itertuples():
        price = heston_price(x0, r, row.strike, rho, sigma, kappa, lra, v0, row.T)
        iv = newton_raphson(S, row.strike, r, row.T, price, option_type)
        heston_prices.append(price)
        heston_ivs.append(iv)
    df = df.copy()
    df["heston_price"] = heston_prices
    df["iv_heston"] = heston_ivs
    df.dropna(subset=["iv_heston"], inplace=True)
    return df.reset_index(drop=True)


def vega_weighted_mse(df, S, r):
    """
    Weights by BSM vega so that IV errors near ATM (high vega, liquid)
    matter more than deep OTM errors, approximating price-space loss.
    Requires columns: strike, T, iv_market, iv_heston.
    """
    vegas = np.array([
        compute_vega(S, row.strike, r, row.iv_market, row.T)
        for row in df.itertuples()
    ])
    iv_err = (df["iv_market"].values - df["iv_heston"].values) ** 2
    return float(np.dot(vegas, iv_err) / vegas.sum())


# Parameter order throughout: (kappa, lra, sigma, rho, v0)
BOUNDS = [
    (0.01, 20.0),   # kappa  — mean-reversion speed
    (0.01, 1.0),    # lra    — long-run variance (theta)
    (0.01, 2.0),    # sigma  — vol of vol
    (-0.99, 0.99),  # rho    — spot/vol correlation
    (0.01, 1.0),    # v0     — initial variance
]


def _objective(params, df, S, r):
    kappa, lra, sigma = params[0], params[1], params[2]
    # Feller condition penalty: keeps variance process away from zero
    feller_violation = max(0.0, sigma**2 - 2 * kappa * lra)
    df_h = compute_heston_ivs(df, S, r, params)
    if df_h.empty:
        return 1e6
    return vega_weighted_mse(df_h, S, r) + 10.0 * feller_violation


def calibrate(df, S, r):
    """
    Two-stage Heston calibration:
      1. Differential Evolution — global search across parameter space.
      2. Nelder-Mead            — local polish of the DE result.

    Returns (params, loss) where params = (kappa, lra, sigma, rho, v0).
    """
    # Subsample for DE: at most 4 expiries, ~10 strikes each
    expiries = df["expiry"].unique()
    step = max(1, len(expiries) // 4)
    sampled_expiries = expiries[::step]
    df_sub = (df[df["expiry"].isin(sampled_expiries)]
              .groupby("expiry", group_keys=False)
              .apply(lambda g: g.iloc[::max(1, len(g) // 10)])
              .reset_index(drop=True))
    print(f"DE subsampled to {len(df_sub)} contracts from {len(df)} total")

    print("Stage 1: Differential Evolution (global search)...")

    def _de_callback(xk, convergence, *args):
        print(f"  convergence={convergence:.6f}  params={np.round(xk, 4)}")

    de_result = differential_evolution(
        _objective,
        bounds=BOUNDS,
        args=(df_sub, S, r),
        strategy="best1bin",
        maxiter=300,
        popsize=12,
        tol=1e-6,
        seed=42,
        polish=False,
        callback=_de_callback,
    )

    print(f"\nStage 2: Nelder-Mead (local polish, full chain of {len(df)} contracts)...")
    nm_result = minimize(
        _objective,
        x0=de_result.x,
        args=(df, S, r),
        method="Nelder-Mead",
        options={"maxiter": 2000, "xatol": 1e-7, "fatol": 1e-7, "disp": True},
    )

    best = nm_result.x if nm_result.fun < de_result.fun else de_result.x
    loss = min(nm_result.fun, de_result.fun)

    kappa, lra, sigma, rho, v0 = best
    print(f"\nCalibrated parameters:")
    print(f"  kappa={kappa:.4f}  lra(theta)={lra:.4f}  sigma={sigma:.4f}"
          f"  rho={rho:.4f}  v0={v0:.4f}")
    print(f"  Feller condition (2κθ >= σ²): {2*kappa*lra:.4f} >= {sigma**2:.4f}"
          f"  {'OK' if 2*kappa*lra >= sigma**2 else 'VIOLATED'}")
    print(f"  Final weighted MSE: {loss:.6f}")

    return tuple(best), loss

def compare_to_bsm(df, S, r):
    """We compare the BSM price to Heston Price. We can sanity check for sigma approaching 0"""
    price_differences = df["bsm_price"] - df["heston_price"]
    df["BSM - HESTON"] = price_differences
    return df.reset_index(drop=True)

def compute_deltas(df, S, r, params):
    kappa, lra, sigma, rho, v0 = params
    h = S * 0.01
    deltas = []
    for row in df.itertuples():
        c_up = heston_price(np.log(S + h), r, row.strike, rho, sigma, kappa, lra, v0, row.T)
        c_dn = heston_price(np.log(S - h), r, row.strike, rho, sigma, kappa, lra, v0, row.T)
        deltas.append((c_up - c_dn) / (2 * h))
    df = df.copy()
    df["delta"] = deltas
    return df.reset_index(drop=True)


def compute_gammas(df, S, r, params):
    kappa, lra, sigma, rho, v0 = params
    h = S * 0.01
    gammas = []
    for row in df.itertuples():
        c_up  = heston_price(np.log(S + h), r, row.strike, rho, sigma, kappa, lra, v0, row.T)
        c_mid = heston_price(np.log(S),     r, row.strike, rho, sigma, kappa, lra, v0, row.T)
        c_dn  = heston_price(np.log(S - h), r, row.strike, rho, sigma, kappa, lra, v0, row.T)
        gammas.append((c_up - 2 * c_mid + c_dn) / h ** 2)
    df = df.copy()
    df["gamma"] = gammas
    return df.reset_index(drop=True)


def compute_vegas(df, S, r, params):
    kappa, lra, sigma, rho, v0 = params
    x0 = np.log(S)
    h = 0.001
    vegas = []
    for row in df.itertuples():
        c_up = heston_price(x0, r, row.strike, rho, sigma, kappa, lra, v0 + h, row.T)
        c_dn = heston_price(x0, r, row.strike, rho, sigma, kappa, lra, v0 - h, row.T)
        vegas.append((c_up - c_dn) / (2 * h))
    df = df.copy()
    df["vega"] = vegas
    return df.reset_index(drop=True)


def compute_thetas(df, S, r, params):
    kappa, lra, sigma, rho, v0 = params
    x0 = np.log(S)
    h = 1 / 252
    thetas = []
    for row in df.itertuples():
        c_up = heston_price(x0, r, row.strike, rho, sigma, kappa, lra, v0, row.T + h)
        c_dn = heston_price(x0, r, row.strike, rho, sigma, kappa, lra, v0, row.T - h)
        thetas.append(-(c_up - c_dn) / (2 * h))
    df = df.copy()
    df["theta"] = thetas
    return df.reset_index(drop=True)


def compute_rhos(df, S, r, params):
    kappa, lra, sigma, rho, v0 = params
    x0 = np.log(S)
    h = 0.001
    rhos = []
    for row in df.itertuples():
        c_up = heston_price(x0, r + h, row.strike, rho, sigma, kappa, lra, v0, row.T)
        c_dn = heston_price(x0, r - h, row.strike, rho, sigma, kappa, lra, v0, row.T)
        rhos.append((c_up - c_dn) / (2 * h))
    df = df.copy()
    df["rho_greek"] = rhos
    return df.reset_index(drop=True)


def compute_greeks(df, S, r, params):
    df = compute_deltas(df, S, r, params)
    df = compute_gammas(df, S, r, params)
    df = compute_vegas(df, S, r, params)
    df = compute_thetas(df, S, r, params)
    df = compute_rhos(df, S, r, params)
    return df


def plot_surfaces(df, ticker=""):
    """
    Left:  3D overlay of market and Heston IV surfaces.
           Vertical gap = calibration error at that (K, T) point.
    Right: 2D heatmap of iv_market - iv_heston.
           Diverging colormap: red = model overestimates, blue = underestimates.
    """
    strikes = df["strike"].values
    mats    = df["T"].values
    iv_mkt  = df["iv_market"].values
    iv_hes  = df["iv_heston"].values
    diff    = iv_mkt - iv_hes

    fig = plt.figure(figsize=(16, 6))
    title = f"{ticker} " if ticker else ""

    # --- Left: overlaid 3D surfaces ---
    ax3d = fig.add_subplot(121, projection="3d")
    ax3d.plot_trisurf(strikes, mats, iv_mkt, cmap="Blues",  alpha=0.65)
    ax3d.plot_trisurf(strikes, mats, iv_hes, cmap="Reds",   alpha=0.65)
    ax3d.set_xlabel("Strike")
    ax3d.set_ylabel("Maturity (yrs)")
    ax3d.set_zlabel("Implied Vol")
    ax3d.set_title(f"{title}IV Surface: Market vs Heston")
    legend_handles = [
        Patch(facecolor="steelblue", alpha=0.7, label="Market"),
        Patch(facecolor="tomato",    alpha=0.7, label="Heston"),
    ]
    ax3d.legend(handles=legend_handles, loc="upper left")

    # --- Right: 2D difference heatmap ---
    ax2d = fig.add_subplot(122)
    sc = ax2d.scatter(strikes, mats, c=diff, cmap="RdBu", s=20,
                      vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
    plt.colorbar(sc, ax=ax2d, label="iv_market − iv_heston")
    ax2d.set_xlabel("Strike")
    ax2d.set_ylabel("Maturity (yrs)")
    ax2d.set_title(f"{title}Calibration Error (IV difference)")

    plt.tight_layout()
    plt.show()


def main():
    ticker = "TSLA"
    r      = 0.05

    # 1. Fetch option chain
    S, df = get_option_chain(ticker, r, min_dte=30, max_dte=365)

    # 2. Compute market IVs (also stores bsm_price for later comparison)
    df = compute_market_ivs(df, S, r)
    print(f"\n{len(df)} contracts with valid market IVs")
    print(df[["expiry", "strike", "T", "mid", "iv_market"]].head(10).to_string(index=False))

    # 3. Calibrate Heston parameters
    params, loss = calibrate(df, S, r)

    # 4. Price every contract with calibrated params to get iv_heston + heston_price
    df = compute_heston_ivs(df, S, r, params)

    # 5. Compare BSM vs Heston prices
    df = compare_to_bsm(df, S, r)
    
    # 6. Compute greeks
    df = compute_greeks(df, S, r, params)
    
    # 7. Plot surfaces
    plot_surfaces(df, ticker=ticker)


if __name__ == "__main__":
    main()
