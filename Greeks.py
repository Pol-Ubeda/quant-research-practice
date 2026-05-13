import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
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

def delta_rebalancing(df, step, K, r, vol, T):
    calls = []
    deltas = []
    prev_delta = 0
    cum_hedge_cost = 0
    hedge_cost = []

    for t in range(0, len(df["Open"]), step):
        if(T-t/252 <= 0): break
        tau = T-t/252

        S = df["Open"].iloc[t]
        c = bsm_price(S,K,r,tau,vol,"call")
        delta = compute_delta(S,K,r,vol,tau)
        calls.append(c)
        deltas.append(delta)

        cum_hedge_cost += (delta - prev_delta) * S
        hedge_cost.append(cum_hedge_cost)

        prev_delta = delta  
    return calls, deltas, hedge_cost

def build_delta_hedged_portfolio(df, vol, K, r, T, step):
    stocks = df["Open"]
    portfolio_value = []
    
    calls, deltas, hedge_cost = delta_rebalancing(df,step,K,r,vol,T)
    time = [t for t in range(len(df["Close"]))]

    calls_repeated = np.repeat(calls,step)
    deltas_repeated= np.repeat(deltas,step)
    hedge_cost_repeated = np.repeat(hedge_cost,step)
    
    for i in range(len(time)):
        portfolio_value.append(calls[0]*np.exp(r*i/252) - hedge_cost_repeated[i] + deltas_repeated[i]*stocks[i] - calls_repeated[i])

    print(f"{'Stock':>10} | {'Call price':>12} | {'Delta':>10} | {'Day':>5} | {'Cumulative Cost (00)':>18} | {'Portfolio Value':>14}")
    for i in range(0, len(time), step):
        print(f"{stocks[i]:>10.2f} | {calls_repeated[i]:>12.4f} | {deltas_repeated[i]:>10.4f} | {time[i]:>5} | {hedge_cost_repeated[i]:>18.2f} | {portfolio_value[i]:>14.2f}")

    n = len(time)
    calls_repeated = calls_repeated[:n]
    deltas_repeated = deltas_repeated[:n]
    hedge_cost_repeated = hedge_cost_repeated[:n]
    
    amount_hedged = stocks[-1] - hedge_cost_repeated[-1]
    print(f"Hedging costs = {hedge_cost_repeated[-1]}")
    print(f"Stock final price = {stocks.iloc[1]}")
    print(f"Amount saved with hedging strategy = {amount_hedged}")

    plt.plot(time, stocks, label='Stock')
    plt.plot(time, portfolio_value, label='Portfolio')
    plt.plot(time, hedge_cost_repeated, label='Hedge')

    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.show()
    
    return

def gamma_rebalancing(df, step, K, r, vol, T, K2, T2):
    # K2, T2 = strike and expiry of the SECOND option used to hedge gamma

    calls, deltas, gammas, option_cost, shares_cost, ws, c2s = [], [], [], [], [], [], []
    cum_option_cost = 0
    cum_share_cost = 0

    for t in range(0, len(df["Open"]), step):
        if T - t/252 <= 0:
            break

        S = df["Open"].iloc[t]
        tau  = max(T  - t/252, 1/252)
        tau2 = max(T2 - t/252, 1/252)
        if tau2 <= 0:
            break

        # Greeks on your short call (position = -1)
        gamma_portfolio = compute_gamma(S, K,  r, vol, tau)
        delta_portfolio = compute_delta(S, K,  r, vol, tau)

        # Greeks on the hedging option (the one you BUY to kill gamma)
        gamma_hedge_option = compute_gamma(S, K2, r, vol, tau2)
        delta_hedge_option = compute_delta(S, K2, r, vol, tau2)

        # Step 1: how many of the hedging option do you need?
        w = -gamma_portfolio / gamma_hedge_option   # units of second option

        # Step 2: net delta after adding w units of the hedging option
        net_delta = delta_portfolio + w * delta_hedge_option

        # Step 3: buy/sell -net_delta shares to flatten delta
        # (shares don't move gamma, so this is safe)
        shares_to_hold = -net_delta

        # Cost of this rebalancing step
        c2 = bsm_price(S, K2, r, tau2, vol, "call")
        delta_w     = w              - (prev_w     if t > 0 else 0)
        delta_share = shares_to_hold - (prev_share if t > 0 else 0)
        cum_option_cost += delta_w * c2
        cum_share_cost += delta_share * S

        calls.append(bsm_price(S, K, r, tau, vol, "call"))
        deltas.append(net_delta)
        option_cost.append(cum_option_cost)
        shares_cost.append(cum_share_cost)
        gammas.append(gamma_portfolio + w * gamma_hedge_option)  # should be ~0
        ws.append(-w)
        c2s.append(c2)
        
        prev_w     = w
        prev_share = shares_to_hold
        
        print(f"t={t:>4} | tau={tau:.4f} | S={S:.2f} | gamma_port={gamma_portfolio:.4f} | gamma_hedge={gamma_hedge_option:.4f} | w={w:.4f} | shares={shares_to_hold:.4f}")

    return calls, deltas, gammas, option_cost, shares_cost, ws, c2s

def build_gamma_hedged_potfolio(df, vol, K, r, T, step):
    stocks = df["Open"]
    stocks = stocks[:int(T*252)]
    portfolio_value = []
    
    #for now we use the same option to kill gamma
    calls, deltas, gammas, option_cost, shares_cost, ws ,c2s = gamma_rebalancing(df,step,K,r,vol,T,K + 5,2 * T)
    time = [t for t in range(int(T*252))]

    calls_repeated = np.repeat(calls,step)
    deltas_repeated = np.repeat(deltas,step)
    gammas_repeated = np.repeat(gammas,step)
    option_cost_repeated = np.repeat(option_cost,step)
    shares_cost_repeated = np.repeat(shares_cost,step)
    ws_repeated = np.repeat(ws,step)
    c2s_repeated = np.repeat(c2s,step)

    for i in range(len(time)):
        portfolio_value.append((calls[0]
                               - calls_repeated[i] 
                               - option_cost_repeated[i] + c2s_repeated[i]*ws_repeated[i]
                               - shares_cost_repeated[i] + deltas_repeated[i]*stocks.iloc[i]))


    n = len(time)
    calls_repeated = calls_repeated[:n]
    deltas_repeated = deltas_repeated[:n]
    gammas_repeated = gammas_repeated[:n]
    option_cost_repeated = option_cost_repeated[:n]
    shares_cost_repeated = shares_cost_repeated[:n]

    plt.plot(time, stocks, label='Stock')
    plt.plot(time, portfolio_value, label='Portfolio')

    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.show()
    return

def vega_rebalancing(df, step, K, r, vol, T, K1, T1, K2, T2):
    # K2, T2 = strike and expiry of the SECOND option used to hedge gamma

    calls, deltas, net_gammas, net_vegas, option1_cost, option2_cost, shares_cost, w1s, w2s, c1s, c2s = [], [], [], [], [], [], [], [], [], [], []
    cum_option1_cost = 0
    cum_option2_cost = 0
    cum_share_cost = 0

    for t in range(0, len(df["Open"]), step):
        if T - t/252 <= 0:
            break

        S = df["Open"].iloc[t]
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
        w1 = (gamma_option2 * vega_portfolio - gamma_portfolio * vega_option2) / (gamma_option1 * vega_option2 - gamma_option2 * vega_option1) # units of second option
        w2 = (-vega_portfolio - w1 * vega_option1) / vega_option2
        
        # Step 2: net delta after adding w units of the hedging option
        net_delta = delta_portfolio + w1 * delta_option1 + w2 * delta_option2

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
        deltas.append(net_delta)
        option1_cost.append(cum_option1_cost)
        option2_cost.append(cum_option2_cost)
        shares_cost.append(cum_share_cost)
        net_gammas.append(gamma_portfolio + w1 * gamma_option1 + w2 * gamma_option2)  # should be ~0
        net_vegas.append(vega_portfolio + w1 * vega_option1 + w2 * vega_option2)
        w1s.append(w1)
        c1s.append(c1)
        w2s.append(w2)
        c2s.append(c2)
        
        prev_w1 = w1
        prev_w2 = w2
        prev_share = shares_to_hold
        
        #print(f"t={t:>4} | tau={tau:.4f} | S={S:.2f} | w1={w1:.4f} | w2={w2:.4f} | net_vega={net_vegas[-1]:.4f} | shares={shares_to_hold:.4f}")

    return calls, deltas, net_gammas, net_vegas, option1_cost, option2_cost, shares_cost, w1s, c1s, w2s, c2s

def build_vega_hedged_portfolio(df, vol, K, r, T, step):
    stocks = df["Open"]
    portfolio_value = []

    K1 = K + 5
    K2 = K + 10
    T1 = 1.5 * T
    T2 = 2.0 * T

    calls, deltas, net_gammas, net_vegas, option1_cost, option2_cost, shares_cost, w1s, c1s, w2s, c2s = vega_rebalancing(df, step, K, r, vol, T, K1, T1, K2, T2)

    time = [t * step for t in range(len(calls))]
    stocks = stocks.iloc[:len(time) * step:step]

    calls_repeated       = np.repeat(calls,         step)[:len(time)]
    deltas_repeated      = np.repeat(deltas,        step)[:len(time)]
    net_gammas_repeated  = np.repeat(net_gammas,    step)[:len(time)]
    net_vegas_repeated   = np.repeat(net_vegas,     step)[:len(time)]
    opt1_cost_repeated   = np.repeat(option1_cost,  step)[:len(time)]
    opt2_cost_repeated   = np.repeat(option2_cost,  step)[:len(time)]
    shares_cost_repeated = np.repeat(shares_cost,   step)[:len(time)]
    w1s_repeated         = np.repeat(w1s,           step)[:len(time)]
    w2s_repeated         = np.repeat(w2s,           step)[:len(time)]
    c1s_repeated         = np.repeat(c1s,           step)[:len(time)]
    c2s_repeated         = np.repeat(c2s,           step)[:len(time)]

    for i in range(len(time)):
        pv = (
            calls[0]
            - calls_repeated[i]
            + (w1s_repeated[i]) * c1s_repeated[i] - opt1_cost_repeated[i]
            + (w2s_repeated[i]) * c2s_repeated[i] - opt2_cost_repeated[i]
            + (-deltas_repeated[i]) * stocks.iloc[i] - shares_cost_repeated[i]
        )
        portfolio_value.append(pv)

    print(f"{'Day':>5} | {'Stock':>10} | {'Call':>10} | {'Net Delta':>10} | {'Net Gamma':>10} | {'Net Vega':>10} | {'w1':>8} | {'w2':>8} | {'Opt1 Cost':>10} | {'Opt2 Cost':>10} | {'Shr Cost':>10} | {'Port Val':>10}")
    print("-" * 130)
    for i in range(0, len(time), step):
        print(
            f"{time[i]:>5} | "
            f"{stocks.iloc[i]:>10.2f} | "
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
    print(f"Final stock price:      {stocks.iloc[-1]:.2f}")
    print(f"Total option1 cost:     {opt1_cost_repeated[-1]:.2f}")
    print(f"Total option2 cost:     {opt2_cost_repeated[-1]:.2f}")
    print(f"Total share cost:       {shares_cost_repeated[-1]:.2f}")

    plt.figure(figsize=(12, 6))
    plt.plot(time, stocks.values,   label='Stock')
    plt.plot(time, portfolio_value, label='Portfolio')
    plt.plot(time, net_gammas_repeated, label='Net Gamma', linestyle='--')
    plt.plot(time, net_vegas_repeated,  label='Net Vega',  linestyle='--')
    plt.legend()
    plt.xlabel("Time (days)")
    plt.ylabel("Value")
    plt.title("Vega-Hedged Portfolio")
    plt.grid(True)
    plt.show()


def main():
    show = "vega"
    df = fetch_ticker('AAPL',"3y","1d")
    vol = estimate_vol(df, "Close")
    S = df["Open"].iloc[0]
    K = S + 200
    r = 0.05
    T = 2
    step = 1 #Step is the daily latency at which we rebalance our portfolio

    if(show == 'delta'): build_delta_hedged_portfolio(df,vol,K,r,T,step)
    elif(show == 'gamma'): build_gamma_hedged_potfolio(df,vol,K,r,T,step)
    elif(show == 'theta'): show_theta(df,vol,S,K,r,T)
    elif(show == 'vega'): build_vega_hedged_portfolio(df,vol,K,r,T,step)    

if __name__ == "__main__":
    main()