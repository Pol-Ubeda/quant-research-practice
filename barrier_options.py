import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

def plot_path(stock):
    Ypoints = stock
    Xpoints = [t for t in  range(len(Ypoints))]
    plt.plot(Xpoints, Ypoints, alpha=0.6)

    
def heston_mc_sim(params: list, r, s0, T, K, barrier: list, barrier_type: str, n_paths, option_type = "call", plot = True, seed = 42):
    """
    Method to simulate and plot paths under Heston model to deduce option price
    params contains heston model parameters, barrier is array containing barrier level (may contain two floats) and barrier_type indicates KO or KI
    """
    steps = int(T * 252)
    dt = 1/252
    rho, kappa, theta, sigma, v0 = params
    prev_v = np.full(n_paths, v0)
    stock = np.full(n_paths, s0)
    option_active = np.full(n_paths, barrier_type == "KO")
    all_paths = [stock.copy()]

    rng = np.random.default_rng(seed)
    Z1 = rng.standard_normal((steps, n_paths))
    Z2 = rho * Z1 + np.sqrt(1 - rho**2) * rng.standard_normal((steps, n_paths))

    for t in range(steps):
        new_v = prev_v + kappa * (theta - prev_v) * dt + sigma * np.sqrt(np.maximum(prev_v, 0) * dt) * Z1[t]
        new_s = stock * np.exp((r - 0.5 * new_v) * dt + np.sqrt(np.maximum(new_v, 0) * dt) * Z2[t])
        stock = new_s
        prev_v = np.maximum(new_v, 0)
        all_paths.append(stock.copy())

        for limit in barrier:
            if limit < s0:
                if barrier_type == "KO":
                    option_active &= new_s >= limit
                else:
                    option_active |= new_s < limit
            if limit > s0:
                if barrier_type == "KO":
                    option_active &= new_s <= limit
                else:
                    option_active |= new_s > limit

    

    if option_type == "call":
        payoffs = np.maximum(0, stock - K)
    else:
        payoffs = np.maximum(0, K - stock)

    prices = np.where(option_active, np.exp(-r * T) * payoffs, 0)
    mc_option_price = np.average(prices)
    if(plot == True):
        for path in np.array(all_paths).T:
            plot_path(path)
        print(mc_option_price)
    return mc_option_price

def compute_greeks(params: list, r, s0, T, K, barrier: list, barrier_type: str, n_paths, option_type = "call", seed = 42):
    h = s0 * 0.01
    true_price = heston_mc_sim(params, r, s0, T, K, barrier, barrier_type, n_paths, option_type, plot = False, seed = seed)
    high_price = heston_mc_sim(params, r, s0 + h, T, K, barrier, barrier_type, n_paths, option_type, plot = False, seed = seed)
    low_price = heston_mc_sim(params, r, s0 - h, T, K, barrier, barrier_type, n_paths, option_type, plot = False, seed = seed)

    delta = (high_price - low_price) / (2 * h)
    gamma = (high_price + low_price - 2*true_price) / (h**2)

    rho, kappa, theta, sigma, v0 = params
    hv = v0 * 0.01
    params_up = [rho, kappa, theta, sigma, v0 + hv]
    params_down = [rho, kappa, theta, sigma, v0 - hv]

    vega_high = heston_mc_sim(params_up, r, s0, T, K, barrier, barrier_type, n_paths, option_type, plot = False, seed = seed)
    vega_low = heston_mc_sim(params_down, r, s0, T, K, barrier, barrier_type, n_paths, option_type, plot = False, seed = seed)

    vega = (vega_high - vega_low) / (2 * hv)
    volga = (vega_high + vega_low - 2*true_price) / (hv**2)

    ht = 1/252
    theta_price = heston_mc_sim(params, r, s0, T - ht, K, barrier, barrier_type, n_paths, option_type, plot = False, seed = seed)
    theta_greek = (theta_price - true_price) / ht

    params_up = [rho, kappa, theta, sigma, v0 + hv]
    params_down = [rho, kappa, theta, sigma, v0 - hv]
    vanna_uu = heston_mc_sim(params_up, r, s0 + h, T, K, barrier, barrier_type, n_paths, option_type, plot = False, seed = seed)
    vanna_ud = heston_mc_sim(params_up, r, s0 - h, T, K, barrier, barrier_type, n_paths, option_type, plot = False, seed = seed)
    vanna_du = heston_mc_sim(params_down, r, s0 + h, T, K, barrier, barrier_type, n_paths, option_type, plot = False, seed = seed)
    vanna_dd = heston_mc_sim(params_down, r, s0 - h, T, K, barrier, barrier_type, n_paths, option_type, plot = False, seed = seed)
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
    T = 0.5
    K = s0 * 1.05
    barrier = [s0 * 0.85]
    barrier_type = "KO"

    heston_mc_sim(params, r, s0, T, K, barrier, barrier_type, n_paths=200, option_type="call")
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.title("Monte Carlo Simulations")
    plt.show()
    
    compute_greeks(params, r, s0, T, K, barrier, barrier_type, n_paths=200, option_type="call")