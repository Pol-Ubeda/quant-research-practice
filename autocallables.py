import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

def plot_path(stock):
    Ypoints = stock
    Xpoints = [t for t in range(len(Ypoints))]
    plt.plot(Xpoints, Ypoints, alpha=0.6)

def heston_mc_sim(params: list, observable_dates: list, r, s0, T, K, autocall_barr: float, coupon: float, coupon_barr: float, put_barr: float, n_paths, principal = 1.0, plot = True, seed = 42):
    """
    Method to simulate and plot paths under Heston model to deduce autocallable price.
    observable_dates are the times (in years) at which autocall/coupon are checked, put_barr is a continuously monitored knock-in barrier
    """
    steps = int(T * 252)
    dt = 1/252
    rho, kappa, theta, sigma, v0 = params
    prev_v = np.full(n_paths, v0)
    stock = np.full(n_paths, s0)
    all_paths = [stock.copy()]

    alive = np.full(n_paths, True)
    put_breached = np.full(n_paths, False)
    pv = np.zeros(n_paths)
    obs_steps = set(int(round(d * 252)) for d in observable_dates)
    obs_steps.add(steps) # adds T to include in coupon payment

    rng = np.random.default_rng(seed)
    Z1 = rng.standard_normal((steps, n_paths))
    Z2 = rho * Z1 + np.sqrt(1 - rho**2) * rng.standard_normal((steps, n_paths))

    for t in range(steps):
        new_v = prev_v + kappa * (theta - prev_v) * dt + sigma * np.sqrt(np.maximum(prev_v, 0) * dt) * Z1[t]
        new_s = stock * np.exp((r - 0.5 * new_v) * dt + np.sqrt(np.maximum(new_v, 0) * dt) * Z2[t])
        stock = new_s
        prev_v = np.maximum(new_v, 0)
        all_paths.append(stock.copy())

        put_breached |= stock <= put_barr

        if t + 1 in obs_steps:
            time = (t + 1) * dt

            pays_coupon = alive & (stock >= coupon_barr)
            pv += np.where(pays_coupon, coupon * np.exp(-r * time), 0)

            autocalled = alive & (stock >= autocall_barr)
            pv += np.where(autocalled, principal * np.exp(-r * time), 0)
            alive &= ~autocalled

    final_payoff = np.where(put_breached, principal * stock / K, principal)
    pv += np.where(alive, final_payoff * np.exp(-r * T), 0)

    mc_price = np.average(pv)
    if(plot == True):
        for path in np.array(all_paths).T:
            plot_path(path)
        print(mc_price)
    return mc_price

def compute_greeks(params: list, observable_dates: list, r, s0, T, K, autocall_barr: float, coupon: float, coupon_barr: float, put_barr: float, n_paths, principal = 1.0, plot = True, seed = 42):
    h = s0 * 0.01
    true_price = heston_mc_sim(params, observable_dates, r, s0, T, K, autocall_barr, coupon, coupon_barr, put_barr, n_paths, principal, plot = False, seed = 42)
    high_price = heston_mc_sim(params, observable_dates, r, s0+h, T, K, autocall_barr, coupon, coupon_barr, put_barr, n_paths, principal, plot = False, seed = 42)
    low_price = heston_mc_sim(params, observable_dates, r, s0-h, T, K, autocall_barr, coupon, coupon_barr, put_barr, n_paths, principal, plot = False, seed = 42)
    delta = (high_price - low_price) / (2 * h)
    gamma = (high_price + low_price - 2*true_price) / (h**2)

    rho, kappa, theta, sigma, v0 = params
    hv = v0 * 0.01
    params_up = [rho, kappa, theta, sigma, v0 + hv]
    params_down = [rho, kappa, theta, sigma, v0 - hv]

    vega_high = heston_mc_sim(params_up, observable_dates, r, s0, T, K, autocall_barr, coupon, coupon_barr, put_barr, n_paths, principal, plot = False, seed = 42)
    vega_low = heston_mc_sim(params_down, observable_dates, r, s0, T, K, autocall_barr, coupon, coupon_barr, put_barr, n_paths, principal, plot = False, seed = 42)

    vega = (vega_high - vega_low) / (2 * hv)
    volga = (vega_high + vega_low - 2*true_price) / (hv**2)

    ht = 1/252
    theta_price = heston_mc_sim(params, observable_dates, r, s0, T-ht, K, autocall_barr, coupon, coupon_barr, put_barr, n_paths, principal, plot = False, seed = 42)
    theta_greek = (theta_price - true_price) / ht

    params_up = [rho, kappa, theta, sigma, v0 + hv]
    params_down = [rho, kappa, theta, sigma, v0 - hv]
    vanna_uu = heston_mc_sim(params_up, observable_dates, r, s0+h, T, K, autocall_barr, coupon, coupon_barr, put_barr, n_paths, principal, plot = False, seed = 42)
    vanna_ud = heston_mc_sim(params_up, observable_dates, r, s0-h, T, K, autocall_barr, coupon, coupon_barr, put_barr, n_paths, principal, plot = False, seed = 42)
    vanna_du = heston_mc_sim(params_down, observable_dates, r, s0+h, T, K, autocall_barr, coupon, coupon_barr, put_barr, n_paths, principal, plot = False, seed = 42)
    vanna_dd = heston_mc_sim(params_down, observable_dates, r, s0-h, T, K, autocall_barr, coupon, coupon_barr, put_barr, n_paths, principal, plot = False, seed = 42)
    vanna = (vanna_uu - vanna_ud - vanna_du + vanna_dd) / (4 * h * hv)

    print(f"Delta: {delta:.4f}, Gamma: {gamma:.4f}, Theta: {theta_greek:.4f}, Vega: {vega:.4f}, Volga: {volga:.4f}, Vanna: {vanna:.4f}")


if __name__ == "__main__":
    ticker = yf.Ticker("AAPL")
    hist = ticker.history(period="1d")
    s0 = hist["Close"].iloc[-1]

    rho = -0.7
    kappa = 2.0
    theta = 0.04
    sigma = 0.3
    v0 = 0.04
    params = [rho, kappa, theta, sigma, v0]

    r = 0.05
    T = 1.0
    K = s0
    observable_dates = [0.25, 0.5, 0.75, 1.0]
    autocall_barr = s0 * 1.0
    coupon = 2.5
    coupon_barr = s0 * 0.8
    put_barr = s0 * 0.7

    heston_mc_sim(params, observable_dates, r, s0, T, K, autocall_barr, coupon, coupon_barr, put_barr, n_paths=200, seed=None)
    compute_greeks(params, observable_dates, r, s0, T, K, autocall_barr, coupon, coupon_barr, put_barr, n_paths=200, seed=42)
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.title("Monte Carlo Simulations")
    plt.show()
