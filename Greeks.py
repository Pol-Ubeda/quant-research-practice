import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import datetime as dt
from scipy.stats import norm

def fetch_ticker(ticker, time_span, interval):
    df = yf.download(
        ticker,
        period=time_span,
        interval = interval,
        auto_adjust = False,
        multi_level_index = False
    )
    return df

def fetch_real_contract(ticker, target_dte_days=365, contract_type="call", moneyness="atm"):
    """
    Fetch a real listed option contract from yfinance.
    
    target_dte_days : roughly how far out you want expiry (e.g. 365 for ~1yr LEAPS)
    moneyness       : "atm", "itm", "otm"
    Returns         : (K, T, expiry_str, iv, contract_row)
    """
    stock = yf.Ticker(ticker)
    S = stock.fast_info["last_price"]
    today = dt.date.today()

    # Pick the expiry closest to target_dte_days
    expiries = stock.options  # tuple of "YYYY-MM-DD" strings
    expiry = min(
        expiries,
        key=lambda e: abs((dt.date.fromisoformat(e) - today).days - target_dte_days)
    )
    expiry_date = dt.date.fromisoformat(expiry)
    T = (expiry_date - today).days / 252  # in years

    chain = stock.option_chain(expiry)
    contracts = chain.calls if contract_type == "call" else chain.puts

    # Pick strike by moneyness
    strikes = contracts["strike"].values
    if moneyness == "atm":
        K = strikes[abs(strikes - S).argmin()]
    elif moneyness == "itm":
        candidates = strikes[strikes < S] if contract_type == "call" else strikes[strikes > S]
        K = candidates[-1] if len(candidates) else strikes[0]
    elif moneyness == "otm":
        candidates = strikes[strikes > S] if contract_type == "call" else strikes[strikes < S]
        K = candidates[0] if len(candidates) else strikes[-1]
    else:
        raise ValueError("moneyness must be 'atm', 'itm', or 'otm'")

    row = contracts[contracts["strike"] == K].iloc[0]
    iv = row["impliedVolatility"]  # market-implied vol — use this instead of historical!

    print(f"\n--- Real contract selected ---")
    print(f"Ticker: {ticker} | S={S:.2f} | Expiry={expiry} | T={T:.3f}yr | K={K} | IV={iv:.2%}")
    print(f"Market price: {row['lastPrice']:.2f} | Bid: {row['bid']:.2f} | Ask: {row['ask']:.2f}")
    print(f"Contract: {row['contractSymbol']}")

    return S, K, T, expiry, iv, row

def simulate_gbm(S, mu, vol, T, n_steps, n_paths):
    dt = T / n_steps
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S
    for t in range(1, n_steps + 1):
        Z = np.random.standard_normal(n_paths)
        paths[:, t] = paths[:, t-1] * np.exp((mu - 0.5*vol**2)*dt + vol*np.sqrt(dt)*Z)
    return paths

def bsm_price(S0, K, r, T, vol, contract_type="call"):
    d1 = (np.log(S0/K) + (r + 0.5*vol**2)*T) / (vol * np.sqrt(T))
    d2 = d1 - (vol * np.sqrt(T))
 
    if contract_type == "call":
        n1 = norm.cdf(d1)
        n2 = norm.cdf(d2)
        price = S0*n1 - (K * np.exp(-r*T) * n2)
    elif contract_type == "put":
        n1 = norm.cdf(-d1)
        n2 = norm.cdf(-d2)
        price = -S0*n1 + (K * np.exp(-r*T) * n2)
    else:
        return None
    return price

def estimate_vol(df, col="Close"):
    prices = df[col]
    log_returns = np.log(prices / prices.shift(1)).dropna()
    s = log_returns.std(ddof=1)
    sigma_hat = s * np.sqrt(252)
    return sigma_hat

def compute_delta(S, K, r, vol, T):
    d1 = (np.log(S/K) + (r + 0.5*vol**2)*T) / (vol * np.sqrt(T))
    return norm.cdf(d1)

def compute_theta(S, K, r, vol, T):
    d1 = (np.log(S/K) + (r + 0.5*vol**2)*T) / (vol * np.sqrt(T))
    d2 = d1 - (vol * np.sqrt(T))
    return -(S * norm.pdf(d1) * vol) / (2*np.sqrt(T)) - (r * K * np.exp(-r*T) * norm.cdf(d2))

def compute_gamma(S, K, r, vol, T):
    d1 = (np.log(S/K) + (r + 0.5*vol**2)*T) / (vol * np.sqrt(T))
    return norm.pdf(d1) / (S * vol * np.sqrt(T))

def show_theta(vol, S, K, r, T):
    theta = compute_theta(S,K,r,vol,T)
    print(f"The theta for this option is {theta}")
    return

def compute_vega(S, K, r, vol, T):
    d1 = (np.log(S/K) + (r + 0.5*vol**2)*T) / (vol * np.sqrt(T))
    return S * np.sqrt(T) * norm.pdf(d1)

def delta_rebalancing(step, K, r, vol, T, gbm_walks):
    calls = []
    deltas = []
    prev_delta = 0
    cum_hedge_cost = 0
    hedge_cost = []

    for t in range(0, int(T * 252), step):
        if(T-t/252 <= 0): break
        tau = T-t/252

        S = gbm_walks[t]
        c = bsm_price(S,K,r,tau,vol,"call")
        delta = compute_delta(S,K,r,vol,tau)
        calls.append(c)
        deltas.append(delta)

        cum_hedge_cost += (delta - prev_delta) * S
        hedge_cost.append(cum_hedge_cost)

        prev_delta = delta  
    return calls, deltas, hedge_cost

def build_delta_hedged_portfolio(vol, K, r, T, step, gbm_walks):
    portfolio_value = []
    
    time = [t for t in range(len(gbm_walks))]
    n = len(time)
    strike = [K for t in range(n)]    
    calls, deltas, hedge_cost = delta_rebalancing(step,K,r,vol,T, gbm_walks)

    calls_repeated = np.repeat(calls,step)
    deltas_repeated= np.repeat(deltas,step)
    hedge_cost_repeated = np.repeat(hedge_cost,step)
    
    for i in range(len(time)):
        portfolio_value.append(calls[0]*np.exp(r*i/252) - hedge_cost_repeated[i] + deltas_repeated[i]*gbm_walks[i] - calls_repeated[i])

    print(f"{'Stock':>10} | {'Call price':>12} | {'Delta':>10} | {'Day':>5} | {'Cumulative Cost (00)':>18} | {'Portfolio Value':>14}")
    for i in range(0, len(time), step):
        print(f"{gbm_walks[i]:>10.2f} | {calls_repeated[i]:>12.4f} | {deltas_repeated[i]:>10.4f} | {time[i]:>5} | {hedge_cost_repeated[i]:>18.2f} | {portfolio_value[i]:>14.2f}")

    n = len(time)
    calls_repeated = calls_repeated[:n]
    deltas_repeated = deltas_repeated[:n]
    hedge_cost_repeated = hedge_cost_repeated[:n]
    
    amount_hedged = gbm_walks[-1] - hedge_cost_repeated[-1]
    print(f"Hedging costs = {hedge_cost_repeated[-1]}")
    print(f"Stock final price = {gbm_walks[-1]}")
    print(f"Amount saved with hedging strategy = {amount_hedged}")

    plt.figure(figsize=(12, 6))
    plt.plot(time, gbm_walks,   label='Stock')
    plt.plot(time, portfolio_value, label='Portfolio')
    plt.plot(time, strike, label='Strike')
    plt.legend()
    plt.xlabel("Time (days)")
    plt.ylabel("Value")
    plt.title("Delta-Hedged Portfolio")
    plt.grid(True)
    plt.show()
    
    return

def gamma_rebalancing(step, K, r, vol, T, K1, T1, gbm_walks):
    # K2, T2 = strike and expiry of the SECOND option used to hedge gamma

    calls, deltas, gammas, option_cost, shares_cost, ws, c2s = [], [], [], [], [], [], []
    cum_option_cost = 0
    cum_share_cost = 0

    for t in range(0, int(T * 252), step):
        if T - t/252 <= 0:
            break

        S = gbm_walks[t]
        tau  = max(T  - t/252, 1/252)
        tau2 = max(T1 - t/252, 1/252)
        if tau2 <= 0:
            break

        # Greeks on your short call (position = -1)
        gamma_portfolio = compute_gamma(S, K,  r, vol, tau)
        delta_portfolio = compute_delta(S, K,  r, vol, tau)

        # Greeks on the hedging option (the one you BUY to kill gamma)
        gamma_hedge_option = compute_gamma(S, K1, r, vol, tau2)
        delta_hedge_option = compute_delta(S, K1, r, vol, tau2)

        # Step 1: how many of the hedging option do you need?
        w = gamma_portfolio / gamma_hedge_option   # units of second option

        # Step 2: net delta after adding w units of the hedging option
        net_delta = delta_portfolio - w * delta_hedge_option

        # Step 3: buy/sell -net_delta shares to flatten delta
        # (shares don't move gamma, so this is safe)
        shares_to_hold = -net_delta

        # Cost of this rebalancing step
        c2 = bsm_price(S, K1, r, tau2, vol, "call")
        delta_w     = w              - (prev_w     if t > 0 else 0)
        delta_share = shares_to_hold - (prev_share if t > 0 else 0)
        cum_option_cost += delta_w * c2
        cum_share_cost += delta_share * S

        calls.append(bsm_price(S, K, r, tau, vol, "call"))
        deltas.append(shares_to_hold)
        option_cost.append(cum_option_cost)
        shares_cost.append(cum_share_cost)
        gammas.append(gamma_portfolio + w * gamma_hedge_option)  # should be ~0
        ws.append(w)
        c2s.append(c2)

        prev_w     = w
        prev_share = shares_to_hold
        
        print(f"t={t:>4} | tau={tau:.4f} | S={S:.2f} | gamma_portfolio={gamma_portfolio:.4f} | gamma_hedge_option={gamma_hedge_option:.4f} | w={w:.4f} | shares={shares_to_hold:.4f}")

    return calls, deltas, gammas, option_cost, shares_cost, ws, c2s

def build_gamma_hedged_potfolio(vol, K, r, T, step, K1, T1, gbm_walks):
    portfolio_value = []
    
    time = [t for t in range(len(gbm_walks))]
    n = len(time)
    strike = [K for t in range(n)]
    
    #for now we use the same option to kill gamma
    calls, deltas, gammas, option_cost, shares_cost, ws ,c2s = gamma_rebalancing(step,K,r,vol,T,K1,T1, gbm_walks)

    calls_repeated = np.repeat(calls,step)
    deltas_repeated = np.repeat(deltas,step)
    gammas_repeated = np.repeat(gammas,step)
    option_cost_repeated = np.repeat(option_cost,step)
    shares_cost_repeated = np.repeat(shares_cost,step)
    ws_repeated = np.repeat(ws,step)
    c2s_repeated = np.repeat(c2s,step)

    for i in range(len(time)):
        portfolio_value.append(
            calls[0]                                    # premium received (1 contract = 100 shares)
            - calls_repeated[i]                         # short call MTM
            + ws_repeated[i] * c2s_repeated[i]         # current value of hedge option position
            - option_cost_repeated[i]                   # cumulative cost of hedge options
            + deltas_repeated[i] * gbm_walks[i]               # share position value (no multiplier)
            - shares_cost_repeated[i]                         # cumulative cash paid for shares
        )
        print(f"t={i} | shares_to_hold={deltas_repeated[i]:.4f} | w={ws_repeated[i]:.2f} | S={gbm_walks[i]:.2f} | shares_cost={shares_cost_repeated[i]:.4f} | share_pnl={deltas_repeated[i]*gbm_walks[i] - shares_cost_repeated[i]:.4f}")
        
    n = len(time)
    calls_repeated = calls_repeated[:n]
    deltas_repeated = deltas_repeated[:n]
    gammas_repeated = gammas_repeated[:n]
    option_cost_repeated = option_cost_repeated[:n]
    shares_cost_repeated = shares_cost_repeated[:n]
    
    # Print the final decomposition for ONE path to see each term
    plt.figure(figsize=(12, 6))
    plt.plot(time, gbm_walks,   label='Stock')
    plt.plot(time, portfolio_value, label='Portfolio')
    plt.plot(time, strike, label='Strike')
    plt.legend()
    plt.xlabel("Time (days)")
    plt.ylabel("Value")
    plt.title("Gamma-Hedged Portfolio")
    plt.grid(True)
    plt.show()
    
    return

def vega_rebalancing(step, K, r, vol, T, K1, T1, K2, T2, gbm_walks):
    # K2, T2 = strike and expiry of the SECOND option used to hedge gamma

    calls, deltas, net_gammas, net_vegas, option1_cost, option2_cost, shares_cost, w1s, w2s, c1s, c2s = [], [], [], [], [], [], [], [], [], [], []
    cum_option1_cost = 0
    cum_option2_cost = 0
    cum_share_cost = 0
    prev_w1, prev_w2, prev_share = 0, 0, 0
    
    for t in range(0, int(T * 252), step):
        if T - t/252 <= 0:
            break

        S = gbm_walks[t]
        tau  = max(T  - t/252, 1/252)
        tau1 = max(T1 - t/252, 1/252)
        tau2 = max(T2 - t/252, 1/252)
        if tau2 <= 0:
            break

        # Greeks on your short call
        vega_portfolio = compute_vega(S, K,  r, vol, tau)
        gamma_portfolio = compute_gamma(S, K,  r, vol, tau)
        delta_portfolio = compute_delta(S, K,  r, vol, tau)

        # Greeks on the hedging option (the one you BUY to kill gamma)
        vega_option1 = compute_vega(S, K1,  r, vol, tau1)
        gamma_option1 = compute_gamma(S, K1, r, vol, tau1)
        delta_option1 = compute_delta(S, K1, r, vol, tau1)
        
        # Greeks on the hedging option (the one you BUY to kill gamma)
        vega_option2 = compute_vega(S, K2,  r, vol, tau2)
        gamma_option2 = compute_gamma(S, K2, r, vol, tau2)
        delta_option2 = compute_delta(S, K2, r, vol, tau2)

        # Step 1: how many of the hedging option do you need for each option? one kills gamma the other vega
        w1 = (-gamma_option2 * vega_portfolio + gamma_portfolio * vega_option2) / (gamma_option1 * vega_option2 - gamma_option2 * vega_option1) # units of second option
        w2 = (vega_portfolio - w1 * vega_option1) / vega_option2
        
        # Step 2: net delta after adding w units of the hedging option
        net_delta = delta_portfolio - w1 * delta_option1 - w2 * delta_option2

        # Step 3: buy/sell -net_delta shares to flatten delta
        # (shares don't move gamma, so this is safe)
        shares_to_hold = -net_delta

        # Cost of this rebalancing step
        c1 = bsm_price(S, K1, r, tau1, vol, "call")
        c2 = bsm_price(S, K2, r, tau2, vol, "call")
        
        delta_w1     = w1             - (prev_w1     if t > 0 else 0)
        delta_w2     = w2             - (prev_w2     if t > 0 else 0)
        delta_share  = shares_to_hold - (prev_share if t > 0 else 0)
        cum_option1_cost += delta_w1 * c1
        cum_option2_cost += delta_w2 * c2
        cum_share_cost += delta_share * S

        calls.append(bsm_price(S, K, r, tau, vol, "call"))
        deltas.append(shares_to_hold)
        option1_cost.append(cum_option1_cost)
        option2_cost.append(cum_option2_cost)
        shares_cost.append(cum_share_cost)
        net_gammas.append(gamma_portfolio + w1 * gamma_option1 + w2 * gamma_option2)  # should be ~0
        net_vegas.append(-vega_portfolio + w1 * vega_option1 + w2 * vega_option2)
        w1s.append(w1)
        c1s.append(c1)
        w2s.append(w2)
        c2s.append(c2)
        
        prev_w1 = w1
        prev_w2 = w2
        prev_share = shares_to_hold
        
        #print(f"t={t:>4} | tau={tau:.4f} | S={S:.2f} | w1={w1:.4f} | w2={w2:.4f} | net_vega={net_vegas[-1]:.4f} | shares={shares_to_hold:.4f}")

    return calls, deltas, net_gammas, net_vegas, option1_cost, option2_cost, shares_cost, w1s, c1s, w2s, c2s

def build_vega_hedged_portfolio(vol, K, r, T, step, K1, T1, K2, T2, gbm_walks):
    portfolio_value = []

    K1 = K + 5
    K2 = K + 10
    T1 = 1.5 * T
    T2 = 2.0 * T

    calls, deltas, net_gammas, net_vegas, option1_cost, option2_cost, shares_cost, w1s, c1s, w2s, c2s = vega_rebalancing(step, K, r, vol, T, K1, T1, K2, T2, gbm_walks)

    time = [t for t in range(len(gbm_walks))]
    n = len(time)
    strike = [K for t in range(n)]

    calls_repeated       = np.repeat(calls,         step)[:n]
    deltas_repeated      = np.repeat(deltas,        step)[:n]
    net_gammas_repeated  = np.repeat(net_gammas,    step)[:n]
    net_vegas_repeated   = np.repeat(net_vegas,     step)[:n]
    opt1_cost_repeated   = np.repeat(option1_cost,  step)[:n]
    opt2_cost_repeated   = np.repeat(option2_cost,  step)[:n]
    shares_cost_repeated = np.repeat(shares_cost,   step)[:n]
    w1s_repeated         = np.repeat(w1s,           step)[:n]
    w2s_repeated         = np.repeat(w2s,           step)[:n]
    c1s_repeated         = np.repeat(c1s,           step)[:n]
    c2s_repeated         = np.repeat(c2s,           step)[:n]

    for i in range(n):
        pv = (
            calls[0]
            - calls_repeated[i]
            + (w1s_repeated[i]) * c1s_repeated[i] - opt1_cost_repeated[i]
            + (w2s_repeated[i]) * c2s_repeated[i] - opt2_cost_repeated[i]
            + (-deltas_repeated[i]) * gbm_walks[i] - shares_cost_repeated[i]
        )
        portfolio_value.append(pv)

    print(f"{'Day':>5} | {'Stock':>10} | {'Call':>10} | {'Net Delta':>10} | {'Net Gamma':>10} | {'Net Vega':>10} | {'w1':>8} | {'w2':>8} | {'Opt1 Cost':>10} | {'Opt2 Cost':>10} | {'Shr Cost':>10} | {'Port Val':>10}")
    print("-" * 130)
    for i in range(0, len(time), step):
        print(
            f"{time[i]:>5} | "
            f"{gbm_walks[i]:>10.2f} | "
            f"{calls_repeated[i]:>10.4f} | "
            f"{deltas_repeated[i]:>10.4f} | "
            f"{net_gammas_repeated[i]:>10.6f} | "
            f"{net_vegas_repeated[i]:>10.6f} | "
            f"{w1s_repeated[i]:>8.4f} | "
            f"{w2s_repeated[i]:>8.4f} | "
            f"{opt1_cost_repeated[i]:>10.2f} | "
            f"{opt2_cost_repeated[i]:>10.2f} | "
            f"{shares_cost_repeated[i]:>10.2f} | "
            f"{portfolio_value[i]:>10.2f}"
        )

    print(f"\nFinal portfolio value:  {portfolio_value[-1]:.2f}")
    print(f"Final stock price:      {gbm_walks[-1]:.2f}")
    print(f"Total option1 cost:     {opt1_cost_repeated[-1]:.2f}")
    print(f"Total option2 cost:     {opt2_cost_repeated[-1]:.2f}")
    print(f"Total share cost:       {shares_cost_repeated[-1]:.2f}")

    plt.figure(figsize=(12, 6))
    plt.plot(time, gbm_walks,   label='Stock')
    plt.plot(time, portfolio_value, label='Portfolio')
    plt.plot(time, net_gammas_repeated, label='Net Gamma', linestyle='--')
    plt.plot(time, net_vegas_repeated,  label='Net Vega',  linestyle='--')
    plt.plot(time, strike, label='Strike')
    plt.legend()
    plt.xlabel("Time (days)")
    plt.ylabel("Value")
    plt.title("Vega-Hedged Portfolio")
    plt.grid(True)
    plt.show()
    
    return

def run_mc_gamma(n_paths, vol, K, r, T, step, K1, T1, S0):
    final_pnls   = []
    final_stocks = []
    all_pnls     = []
    n_days       = int(T * 252)

    for i in range(n_paths):
        # 1. Simulate one GBM path (reusing your existing function)
        path = simulate_gbm(S0, r, vol, T, n_days - 1, 1)[0]  # shape: (n_days,)

        # 2. Run gamma rebalancing
        try:
            calls, deltas, gammas, option_cost, shares_cost, ws, c2s = \
                gamma_rebalancing(step, K, r, vol, T, K1, T1, path)
        except Exception as e:
            print(f"Path {i} failed: {e}")
            continue

        n = len(path)

        # 3. Repeat rebalancing-step values to daily resolution
        calls_r        = repeat_to_length(calls,        step, n)
        deltas_r       = repeat_to_length(deltas,       step, n)   # shares_to_hold = -net_delta (already fixed in your code)
        option_cost_r  = repeat_to_length(option_cost,  step, n)
        shares_cost_r  = repeat_to_length(shares_cost,  step, n)
        ws_r           = repeat_to_length(ws,           step, n)
        c2s_r          = repeat_to_length(c2s,          step, n)

        # 4. Compute daily P&L series
        pnl_series = (
            calls[0]                          # premium received upfront
            - calls_r                         # short call MTM
            + ws_r * c2s_r                    # hedge option position MTM
            - option_cost_r                   # cumulative cost of hedge options
            + deltas_r * path[:n]             # share position MTM  (deltas_r = shares_to_hold)
            - shares_cost_r                   # cumulative cash paid for shares
        )

        final_pnls.append(pnl_series[-1])
        final_stocks.append(path[-1])
        all_pnls.append(pnl_series)

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n_paths} paths done...")

    final_pnls   = np.array(final_pnls)
    final_stocks = np.array(final_stocks)

    # Print the final decomposition for ONE path to see each term
    print(f"\n--- Final P&L decomposition (t=-1) ---")
    i = n - 1
    print(f"Premium received      : {calls[0]:.4f}")
    print(f"Short call MTM        : {-calls_r[i]:.4f}")
    print(f"Hedge option MTM      : {ws_r[i] * c2s_r[i]:.4f}")
    print(f"Hedge option cost     : {-option_cost_r[i]:.4f}")
    print(f"Share position MTM    : {deltas_r[i] * path[i]:.4f}")
    print(f"Share cost            : {-shares_cost_r[i]:.4f}")
    print(f"Total                 : {pnl_series[i]:.4f}")
    # Pad paths to same length (some may terminate early due to tau2 <= 0)
    max_len  = max(len(p) for p in all_pnls)
    all_pnls = np.array([np.pad(p, (0, max_len - len(p)), constant_values=p[-1]) for p in all_pnls])

    # --- Plots ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Histogram of final P&L
    axes[0].hist(final_pnls, bins=50, edgecolor='black', color='steelblue')
    axes[0].axvline(np.mean(final_pnls), color='red',    linestyle='--', label=f'Mean = {np.mean(final_pnls):.2f}')
    axes[0].axvline(calls[0],            color='green',  linestyle='--', label=f'Premium = {calls[0]:.2f}')
    axes[0].set_title("Distribution of Final P&L")
    axes[0].set_xlabel("P&L")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    # Plot 2: Final P&L vs final stock price — KEY diagnostic
    axes[1].scatter(final_stocks, final_pnls, alpha=0.3, s=10, color='steelblue')
    axes[1].axhline(np.mean(final_pnls), color='red',   linestyle='--', label=f'Mean P&L = {np.mean(final_pnls):.2f}')
    axes[1].axvline(K,                   color='green', linestyle='--', label=f'Strike K = {K}')
    axes[1].set_title("Final P&L vs Final Stock Price\n(should be flat horizontal cloud)")
    axes[1].set_xlabel("Final Stock Price $S_T$")
    axes[1].set_ylabel("Final P&L")
    axes[1].legend()

    # Plot 3: Fan chart across all paths
    pcts = np.percentile(all_pnls, [10, 25, 50, 75, 90], axis=0)
    t_axis = np.arange(all_pnls.shape[1])
    axes[2].fill_between(t_axis, pcts[0], pcts[4], alpha=0.15, color='steelblue', label='10-90th pct')
    axes[2].fill_between(t_axis, pcts[1], pcts[3], alpha=0.30, color='steelblue', label='25-75th pct')
    axes[2].plot(t_axis, pcts[2], color='black', linewidth=1.5, label='Median')
    axes[2].axhline(0, color='red', linestyle='--')
    axes[2].set_title("P&L Fan Chart (all paths)")
    axes[2].set_xlabel("Time (days)")
    axes[2].set_ylabel("P&L")
    axes[2].legend()

    plt.suptitle(f"Gamma Hedge MC — {n_paths} paths | step={step} | K={K} | K1={K1}", fontsize=13)
    plt.tight_layout()
    plt.show()

    # --- Summary stats ---
    print(f"\n{'='*45}")
    print(f"MC Results — {n_paths} paths, step={step}")
    print(f"{'='*45}")
    print(f"Initial premium : {calls[0]:.4f}")
    print(f"Mean final P&L  : {np.mean(final_pnls):.4f}")
    print(f"Std final P&L   : {np.std(final_pnls):.4f}")
    print(f"Sharpe (P&L)    : {np.mean(final_pnls)/np.std(final_pnls):.4f}")
    print(f"Min / Max       : {np.min(final_pnls):.4f} / {np.max(final_pnls):.4f}")
    print(f"% paths > 0     : {100*np.mean(final_pnls > 0):.1f}%")

    return final_pnls, final_stocks, all_pnls

def diagnose_gamma_hedge(vol, K, r, T, K1, T1, S):
    tau  = T
    tau1 = T1
    
    g_port  = compute_gamma(S, K,  r, vol, tau)
    d_port  = compute_delta(S, K,  r, vol, tau)
    
    g_hedge = compute_gamma(S, K1, r, vol, tau1)
    d_hedge = compute_delta(S, K1, r, vol, tau1)
    
    w = -g_port / g_hedge
    net_delta = d_port + w * d_hedge
    shares_to_hold = -net_delta
    
    print(f"\n\nS={S:.2f} | K={K} | K1={K1} | T={T:.3f} | T1={T1:.3f} | vol={vol:.4f}")
    print(f"gamma_port={g_port:.6f}  |  gamma_hedge={g_hedge:.6f}")
    print(f"delta_port={d_port:.6f}  |  delta_hedge={d_hedge:.6f}")
    print(f"w={w:.4f}  |  net_delta={net_delta:.6f}  |  shares_to_hold={shares_to_hold:.6f}")
    print(f"net_gamma = {g_port + w * g_hedge:.8f}  <-- should be ~0")
    print(f"c1 (short)  = {bsm_price(S, K,  r, tau,  vol, 'call'):.4f}")
    print(f"c2 (hedge)  = {bsm_price(S, K1, r, tau1, vol, 'call'):.4f}")

def repeat_to_length(arr, step, n):
    """Repeat rebalancing values to daily resolution, 
    filling the tail with the last known value (not zeros)."""
    repeated = np.repeat(arr, step)
    if len(repeated) >= n:
        return repeated[:n]
    else:
        # pad with last value, not zeros
        pad = n - len(repeated)
        return np.concatenate([repeated, np.full(pad, repeated[-1])])
    
    
def main():
    show = "vega"
    ticker = 'AAPL'
    df = fetch_ticker(ticker,"3y","1d")
    #vol = estimate_vol(df, "Close")
    
    S, K, T, expiry, market_iv, contract = fetch_real_contract(ticker, target_dte_days=365, contract_type="call", moneyness="atm")
    vol = market_iv
    S, K1, T1, expiry, market_iv, contract = fetch_real_contract(ticker, target_dte_days=600, contract_type="call", moneyness="itm")
    S, K2, T2, expiry, market_iv, contract = fetch_real_contract(ticker, target_dte_days=420, contract_type="call", moneyness="otm")

    
    r = 0.045
    step = 1 #Step is the daily latency at which we rebalance our portfolio
    n_steps = int(T*252) - 1
    gbm_walks = simulate_gbm(S,r,vol,T,n_steps,1)

    if(show == 'delta'): build_delta_hedged_portfolio(vol,K,r,T,step, gbm_walks[0])
    #elif(show == 'gamma'): build_gamma_hedged_potfolio(vol,K,r,T,step, K1, T1, gbm_walks[0])
    #elif(show == 'theta'): show_theta(vol,S,K,r,T)
    elif(show == 'vega'): build_vega_hedged_portfolio(vol,K,r,T,step, K1, T1, K2, T2, gbm_walks[0])
    
    #diagnose_gamma_hedge(vol, K, r, T, K1+30, T1, S)
    #run_mc_gamma(n_paths=500, vol=vol, K=K, r=r, T=T, step=step, K1=K1+30, T1=T1, S0=S)

if __name__ == "__main__":
    main()